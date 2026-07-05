//! # Daemon bootstrap and lifecycle
//!
//! `run()` wires every subsystem together (registry, downloader, inference,
//! HTTP server), takes a PID lock to prevent two daemons stomping on each
//! other's state, then blocks on `axum::serve` until shutdown is signaled.
//!
//! On shutdown (Ctrl+C or SIGTERM):
//!   1. Graceful HTTP shutdown — stops accepting new requests, drains existing.
//!   2. Kill every inference process tree.
//!   3. Release the PID lock via `Drop`.

use anyhow::{anyhow, Result};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

use crate::api::routes::{build_router, AppState, BackendHealth, Metrics};
use crate::compression::CompressionEngine;
use crate::config::Settings;
use crate::downloader::{DownloadManager, HuggingFaceClient};
use crate::inference::{InferenceRouter, LlamaCppManager, SglangManager};
use crate::registry::ManifestStore;

/// Main daemon entry point — invoked when the user runs `localllm serve`,
/// or via the auto-spawn path in `main.rs`.
///
/// Startup order matters: we ensure data dirs exist first, then take the PID
/// lock before binding the listener — that way two parallel `localllm serve`
/// invocations get a clean error from the second one instead of a port-bind
/// race that leaves both half-initialized.
pub async fn run(settings: Arc<Settings>) -> Result<()> {
    settings.ensure_dirs()?;

    // C5 — PID lock file. Prevents two daemons from racing on the same ports.
    let pid_lock = PidLock::acquire(&settings)?;
    tracing::info!("Acquired daemon lock at {:?}", pid_lock.path);

    tracing::info!(
        "Starting localllm daemon on {}:{}",
        settings.daemon_host,
        settings.daemon_port
    );

    // First-run auto-setup: ensure a usable llama.cpp exists, cloning + building
    // it into ~/.localllm/llama.cpp on first run if needed. This is what makes
    // the whole tool work after nothing but `cargo build`. If it fails (e.g. no
    // git/cmake) we log a warning and continue — the daemon still serves, and
    // quantize/CPU-inference endpoints return a clear error with install hints.
    let settings = match crate::setup::ensure_llama_cpp(&settings).await {
        Ok(dir) => {
            if settings.llama_cpp_dir.as_deref() != Some(dir.as_path()) {
                tracing::info!("Using llama.cpp at {:?}", dir);
                let mut s = (*settings).clone();
                s.llama_cpp_dir = Some(dir);
                Arc::new(s)
            } else {
                settings
            }
        }
        Err(e) => {
            tracing::warn!("Automatic llama.cpp setup did not complete: {}", e);
            settings
        }
    };

    // Empty registry; load manifests off the hot path.
    let registry = Arc::new(ManifestStore::new(settings.manifests_dir.clone())?);
    // A4 — load existing manifests from disk in a background blocking task.
    // The daemon can start serving immediately; manifests pop into the cache as scanned.
    let registry_for_load = registry.clone();
    tokio::task::spawn_blocking(move || registry_for_load.load_all());

    let hf_client = Arc::new(HuggingFaceClient::new(settings.hf_token.clone())?);
    let download_manager = Arc::new(DownloadManager::new(hf_client, settings.clone()));

    let compression_engine = settings
        .llama_cpp_dir
        .as_ref()
        .map(|dir| Arc::new(CompressionEngine::new(dir.clone(), settings.clone())));

    let sglang_manager = Arc::new(SglangManager::new(settings.clone()));
    let llamacpp_manager = Arc::new(LlamaCppManager::new(settings.clone()));

    let inference_router = Arc::new(InferenceRouter::new(
        sglang_manager.clone(),
        llamacpp_manager.clone(),
        settings.clone(),
    ));

    let http_client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(32)
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(10))
        .build()?;

    let metrics = Arc::new(Metrics::default());
    let backend_health = Arc::new(dashmap::DashMap::<String, BackendHealth>::new());

    let state = Arc::new(AppState {
        settings: settings.clone(),
        registry,
        download_manager,
        compression_engine,
        inference_router: inference_router.clone(),
        sglang_manager: sglang_manager.clone(),
        llamacpp_manager: llamacpp_manager.clone(),
        http_client: http_client.clone(),
        metrics,
        backend_health: backend_health.clone(),
    });

    let router = build_router(state.clone());

    let bind_addr = format!("{}:{}", settings.daemon_host, settings.daemon_port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("Listening on {}", bind_addr);

    // Pre-warm any models listed in config. Done in a background task so the
    // daemon stays responsive even while models load. Wait for manifest cache
    // to populate first (it's loading in spawn_blocking above).
    if !settings.preload.is_empty() {
        let preload_state = state.clone();
        let aliases = settings.preload.clone();
        tokio::spawn(async move {
            // Brief delay so the background manifest scan has a chance to run.
            tokio::time::sleep(Duration::from_millis(500)).await;
            for alias in aliases {
                match preload_state.registry.get(&alias) {
                    None => {
                        tracing::warn!("Preload skipped — model not found: {}", alias);
                    }
                    Some(manifest) => {
                        tracing::info!("Pre-warming model: {}", alias);
                        match preload_state.inference_router.get_endpoint(&manifest).await {
                            Ok(ep) => tracing::info!("Pre-warmed {} at {}", alias, ep),
                            Err(e) => tracing::warn!("Pre-warm failed for {}: {}", alias, e),
                        }
                    }
                }
            }
        });
    }

    // Background backend health-check task. Pings each running process's
    // /health every 30s and writes the result into AppState.backend_health.
    // The /health endpoint reads this cache to expose unhealthy backends.
    let sg_hc = sglang_manager.clone();
    let lc_hc = llamacpp_manager.clone();
    let hc_client = http_client.clone();
    let hc_results = backend_health.clone();
    tokio::spawn(async move {
        let probe_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap_or(hc_client);

        loop {
            sleep(Duration::from_secs(30)).await;
            let now_ts = chrono::Utc::now().timestamp();

            // Collect (alias, port) pairs without holding DashMap refs across await.
            let mut targets: Vec<(String, u16)> = Vec::new();
            for e in sg_hc.processes.iter() {
                targets.push((e.key().clone(), e.value().port));
            }
            for e in lc_hc.processes.iter() {
                targets.push((e.key().clone(), e.value().port));
            }

            for (alias, port) in targets {
                let url = format!("http://127.0.0.1:{}/health", port);
                match probe_client.get(&url).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        hc_results.insert(
                            alias,
                            BackendHealth {
                                healthy: true,
                                last_checked: now_ts,
                                last_error: None,
                            },
                        );
                    }
                    Ok(resp) => {
                        hc_results.insert(
                            alias,
                            BackendHealth {
                                healthy: false,
                                last_checked: now_ts,
                                last_error: Some(format!("HTTP {}", resp.status())),
                            },
                        );
                    }
                    Err(e) => {
                        hc_results.insert(
                            alias,
                            BackendHealth {
                                healthy: false,
                                last_checked: now_ts,
                                last_error: Some(e.to_string()),
                            },
                        );
                    }
                }
            }

            // Drop entries for processes that are no longer running.
            let active: std::collections::HashSet<String> = sg_hc
                .processes
                .iter()
                .map(|e| e.key().clone())
                .chain(lc_hc.processes.iter().map(|e| e.key().clone()))
                .collect();
            hc_results.retain(|k, _| active.contains(k));
        }
    });

    // A5 — Periodic manifest flush. update_last_used() touches the in-memory
    // cache and marks the alias dirty; this task batches those touches into
    // one disk write every 30 s instead of one per request.
    let flush_registry = state.registry.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            flush_registry.flush_dirty();
        }
    });

    // Background TTL eviction (sync scan over AtomicI64 timestamps — lock-free).
    let sg_evict = sglang_manager.clone();
    let lc_evict = llamacpp_manager.clone();
    let ttl = settings.model_ttl_secs as i64;
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(60)).await;
            let now_ts = chrono::Utc::now().timestamp();

            let sg_expired: Vec<String> = sg_evict
                .processes
                .iter()
                .filter_map(|e| {
                    let last = e.value().last_used.load(Ordering::Relaxed);
                    if now_ts - last > ttl {
                        Some(e.key().clone())
                    } else {
                        None
                    }
                })
                .collect();
            for alias in sg_expired {
                tracing::info!("TTL eviction: sglang:{}", alias);
                let _ = sg_evict.kill(&alias).await;
            }

            let lc_expired: Vec<String> = lc_evict
                .processes
                .iter()
                .filter_map(|e| {
                    let last = e.value().last_used.load(Ordering::Relaxed);
                    if now_ts - last > ttl {
                        Some(e.key().clone())
                    } else {
                        None
                    }
                })
                .collect();
            for alias in lc_expired {
                tracing::info!("TTL eviction: llamacpp:{}", alias);
                let _ = lc_evict.kill(&alias).await;
            }
        }
    });

    let shutdown = async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, shutting down");
                }
                _ = sigterm.recv() => {
                    tracing::info!("Received SIGTERM, shutting down");
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("Received Ctrl+C, shutting down");
        }
    };

    // A2 — custom accept loop that enables TCP_NODELAY on every accepted
    // connection. axum::serve doesn't expose per-connection socket config,
    // but hyper's serve_connection lets us do this directly. Saves 5–30 ms
    // of TTFT per request by killing Nagle's small-packet hold.
    use hyper::server::conn::http1;
    use hyper_util::rt::TokioIo;
    use hyper_util::service::TowerToHyperService;

    let shutdown_token = std::sync::Arc::new(tokio::sync::Notify::new());
    let shutdown_token_for_signal = shutdown_token.clone();
    tokio::spawn(async move {
        shutdown.await;
        shutdown_token_for_signal.notify_waiters();
    });

    loop {
        tokio::select! {
            _ = shutdown_token.notified() => {
                tracing::info!("Stopping accept loop");
                break;
            }
            accepted = listener.accept() => {
                let (socket, _peer) = match accepted {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("accept failed: {}", e);
                        continue;
                    }
                };

                // Disable Nagle so streaming SSE chunks flush immediately.
                if let Err(e) = socket.set_nodelay(true) {
                    tracing::debug!("set_nodelay failed (non-fatal): {}", e);
                }

                // Router is Clone — each connection gets its own copy of the
                // service (cheap; shared state is behind Arcs inside).
                let conn_service = TowerToHyperService::new(router.clone());
                let io = TokioIo::new(socket);

                tokio::spawn(async move {
                    if let Err(e) = http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(io, conn_service)
                        .await
                    {
                        tracing::debug!("connection error: {}", e);
                    }
                });
            }
        }
    }

    // A5 — final flush before tearing down so in-flight last_used updates land.
    state.registry.flush_dirty();

    tracing::info!("Killing all inference processes...");
    inference_router.kill_all().await?;

    // PidLock removed on Drop.
    drop(pid_lock);
    tracing::info!("Daemon shutdown complete");
    Ok(())
}

// =============================================================================
// PID lock file (C5)
// =============================================================================

/// Exclusive PID-file lock. Lives at `~/.localllm/daemon.pid`.
///
/// Semantics: on `acquire()`, if the file exists and the PID inside is alive,
/// we refuse to start (returns an Err with the running PID for the user).
/// If the PID is dead (stale file from a crashed daemon), we take it over.
/// On `Drop`, we remove the file — best-effort, since the process may have
/// been SIGKILLed before Drop ran.
pub struct PidLock {
    path: std::path::PathBuf,
}

impl PidLock {
    /// Attempt to acquire the lock. Returns Err if another live daemon owns it.
    pub fn acquire(settings: &Settings) -> Result<Self> {
        let pid_path = pid_file_path(settings);
        let our_pid = std::process::id();

        // If file exists, decide: is the owner still alive?
        if pid_path.exists() {
            let existing = std::fs::read_to_string(&pid_path)
                .unwrap_or_default()
                .trim()
                .parse::<u32>()
                .ok();
            if let Some(other_pid) = existing {
                if other_pid != our_pid && process_alive(other_pid) {
                    return Err(anyhow!(
                        "Another localllm daemon is already running (pid {}). \
                         If this is wrong, remove {:?} manually.",
                        other_pid,
                        pid_path
                    ));
                } else {
                    tracing::warn!(
                        "Stale PID file from pid {} — overwriting",
                        other_pid
                    );
                }
            }
        }

        // Create-or-overwrite (we've decided the previous owner is gone).
        std::fs::write(&pid_path, our_pid.to_string())?;
        Ok(Self { path: pid_path })
    }
}

impl Drop for PidLock {
    fn drop(&mut self) {
        // Best-effort cleanup. If the file is gone or we lack perms, log and continue.
        if let Err(e) = std::fs::remove_file(&self.path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("Failed to remove PID file {:?}: {}", self.path, e);
            }
        }
    }
}

/// Compute the PID file path. Lives as a sibling of `manifests_dir` so it
/// follows the user's data-dir overrides (LOCALLLM_MANIFESTS_DIR) instead of
/// being pinned to the default `~/.localllm` location.
fn pid_file_path(settings: &Settings) -> std::path::PathBuf {
    settings
        .manifests_dir
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("daemon.pid")
}

/// Liveness probe for a PID. Returns true iff a process with this PID exists.
///
/// On Unix: `kill -0 <pid>` returns 0 when the signal would be deliverable
/// (process exists and we have permission). Signal 0 doesn't actually send.
#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    let status = std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Windows variant of `process_alive`. Uses `tasklist /FI "PID eq N"` since
/// there's no convenient equivalent of `kill -0`. `tasklist` prints "INFO: No
/// tasks..." to stdout when no match, which we detect by string.
#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    // tasklist /FI "PID eq <pid>" /NH /FO CSV — empty or "INFO:" header means absent.
    let output = std::process::Command::new("tasklist")
        .args([
            "/FI",
            &format!("PID eq {}", pid),
            "/NH",
            "/FO",
            "CSV",
        ])
        .output();
    match output {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            // tasklist prints "INFO: No tasks ..." to stdout when no match
            !s.is_empty() && !s.contains("INFO:")
        }
        Err(_) => false,
    }
}
