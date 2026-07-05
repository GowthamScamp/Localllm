//! # sglang backend manager
//!
//! Manages one `python -m sglang.launch_server` child process per model alias.
//! `DashMap` keyed by alias gives us shard-level concurrency without an
//! Arc<Mutex<HashMap<...>>>. The OS process is owned via `tokio::process::Child`,
//! lifetime is tied to the entry in the DashMap.
//!
//! Three subtleties worth knowing:
//!   * `last_used` is `AtomicI64` (Unix seconds), not `Mutex<DateTime>`. This
//!     means the eviction loop can scan ages without grabbing any locks.
//!   * `port_alloc_lock` serializes free-port discovery to prevent two
//!     concurrent spawns from picking the same port.
//!   * Child processes go in a new process group / Windows job so `kill()`
//!     takes down the whole tree, including sglang's Python worker subprocs.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

use crate::config::Settings;
use crate::gpu::VramManager;
use crate::inference::logbuf::LogRingBuffer;
use crate::registry::ModelManifest;

/// One running sglang inference process. Owned by `SglangManager.processes`.
/// Drop semantics: when removed from the DashMap, `child` is killed when its
/// `tokio::process::Child` Drop runs — but we explicitly kill the whole tree
/// first via `kill_process_tree` to be sure Python workers die too.
pub struct SglangProcess {
    pub model_alias: String,
    pub port: u16,
    pub model_path: PathBuf,
    pub child: Child,
    pub started_at: DateTime<Utc>,
    /// Last-used Unix-second timestamp. `Arc<AtomicI64>` lets us hand a
    /// touch-only handle to streaming proxies without aliasing the whole
    /// `SglangProcess` struct.
    pub last_used: Arc<AtomicI64>,
    /// Recent stdout/stderr lines. Reader tasks push here; `/api/logs/<alias>`
    /// reads a snapshot. Capped at LogRingBuffer::DEFAULT_CAPACITY lines.
    pub logs: Arc<LogRingBuffer>,
}

/// Per-alias registry of running sglang processes plus a lock for port allocation.
/// Cloning is not supported; share via `Arc<SglangManager>`.
pub struct SglangManager {
    pub processes: DashMap<String, SglangProcess>,
    pub settings: Arc<Settings>,
    /// Serializes free-port discovery. Held briefly; never crosses an await
    /// boundary that does I/O, so it never blocks the runtime.
    port_alloc_lock: Mutex<()>,
    /// A3 — coalesces concurrent cold-spawn requests. When the first request
    /// for an alias arrives and the model isn't loaded, an entry is inserted
    /// with an `Arc<Notify>`. Subsequent requests await that Notify instead of
    /// spawning their own redundant backend. Saves duplicate Python/CUDA load.
    spawn_in_flight: Mutex<std::collections::HashMap<String, Arc<tokio::sync::Notify>>>,
}

impl SglangManager {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self {
            processes: DashMap::new(),
            settings,
            port_alloc_lock: Mutex::new(()),
            spawn_in_flight: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Spawn a fresh sglang server for this manifest. Returns the port the
    /// server is listening on once `/health` returns 200.
    ///
    /// Steps:
    ///   1. Allocate a free port (under `port_alloc_lock`).
    ///   2. Build the `python -m sglang.launch_server` command with model path,
    ///      bind addr, dtype, and VRAM headroom (`--mem-fraction-static 0.85`).
    ///   3. Spawn in a new process group so kill can take down workers.
    ///   4. Capture stdout/stderr to tracing::debug at the line level.
    ///   5. Poll `/health` with exponential backoff (100ms → 2s cap) until
    ///      success OR `sglang_startup_timeout_secs` elapses (default 120s).
    ///   6. Insert into the processes map.
    pub async fn spawn(&self, manifest: &ModelManifest) -> Result<u16> {
        let port = {
            let _lock = self.port_alloc_lock.lock().await;
            find_free_port(
                self.settings.sglang_port_range_start,
                self.settings.sglang_port_range_end,
            )?
        };

        let model_path = manifest
            .gguf_path
            .clone()
            .unwrap_or_else(|| manifest.local_path.clone());

        tracing::info!(
            "Spawning sglang for {} on port {} with model {:?}",
            manifest.alias,
            port,
            model_path
        );

        // B1 — tensor parallelism across all visible GPUs. >1 GPU → shard the
        // model with --tp-size N (the main multi-GPU throughput lever). Single
        // GPU omits the flag (sglang defaults to tp=1).
        let gpu_count = VramManager::query_gpus().map(|g| g.len()).unwrap_or(0);

        // Resolve Python interpreter for this OS (python3 / python / py).
        let python = crate::platform::python_command();

        let mut cmd = Command::new(&python);
        cmd.args(["-m", "sglang.launch_server"])
            .arg("--model-path")
            .arg(&model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--dtype")
            .arg("bfloat16")
            // B4 — configurable static VRAM reservation (was hardcoded 0.85).
            .arg("--mem-fraction-static")
            .arg(format!("{:.2}", self.settings.sglang_mem_fraction))
            // B2 — chunked prefill keeps the scheduler responsive on long prompts.
            .arg("--chunked-prefill-size")
            .arg(self.settings.sglang_chunked_prefill_size.to_string());

        // B1 — multi-GPU tensor parallelism.
        if gpu_count > 1 {
            tracing::info!(
                "sglang {}: tensor-parallel across {} GPUs",
                manifest.alias,
                gpu_count
            );
            cmd.arg("--tp-size").arg(gpu_count.to_string());
        }

        // B3 — Torch compile: steady-state speedup at the cost of a slow first
        // spawn. Opt-in only.
        if self.settings.sglang_torch_compile {
            cmd.arg("--enable-torch-compile");
        }

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // A8 — put the child in its own process group so kill() cleans up
        // grandchildren too (sglang spawns Python worker processes).
        configure_process_group(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn sglang: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture sglang stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("Failed to capture sglang stderr"))?;

        let alias_stdout = manifest.alias.clone();
        let alias_stderr = manifest.alias.clone();

        // Create the log buffer up-front so both reader tasks can push into it.
        let logs = Arc::new(LogRingBuffer::default());
        let logs_stdout = Arc::clone(&logs);
        let logs_stderr = Arc::clone(&logs);

        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::debug!("[sglang:{}:stdout] {}", alias_stdout, line);
                logs_stdout.push(format!("[stdout] {}", line));
            }
        });

        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::debug!("[sglang:{}:stderr] {}", alias_stderr, line);
                logs_stderr.push(format!("[stderr] {}", line));
            }
        });

        // Health-check polling with exponential backoff (100ms → 200ms → ... → 2s).
        // This makes models that load fast (e.g. small quants) become available
        // ~10-20x sooner than a fixed 2s polling interval.
        let health_url = format!("http://127.0.0.1:{}/health", port);
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .timeout(Duration::from_secs(5))
            .pool_idle_timeout(Duration::from_secs(30))
            .tcp_nodelay(true)
            .build()
            .unwrap_or_default();

        let timeout_total_ms = self.settings.sglang_startup_timeout_secs * 1000;
        let mut elapsed_ms: u64 = 0;
        let mut delay_ms: u64 = 100;

        loop {
            sleep(Duration::from_millis(delay_ms)).await;
            elapsed_ms = elapsed_ms.saturating_add(delay_ms);

            if let Ok(resp) = client.get(&health_url).send().await {
                if resp.status().is_success() {
                    tracing::info!(
                        "sglang for {} ready on port {} after {}ms",
                        manifest.alias,
                        port,
                        elapsed_ms
                    );
                    break;
                }
            }

            if elapsed_ms >= timeout_total_ms {
                let _ = child.kill().await;
                return Err(anyhow!(
                    "sglang startup timeout after {}s for {}",
                    self.settings.sglang_startup_timeout_secs,
                    manifest.alias
                ));
            }

            // Cap exponential backoff at 2s
            delay_ms = (delay_ms.saturating_mul(2)).min(2000);
        }

        // A1 — warm-up. Send a 1-token completion so the backend JIT-compiles
        // CUDA kernels and pre-fills the KV cache. Saves 2–5x latency on the
        // *real* first user request. Best-effort: failures are logged but
        // don't block readiness, since the model is healthy enough for /health.
        warmup_backend(&client, port, &manifest.alias, "sglang").await;

        let now = Utc::now();
        let now_ts = now.timestamp(); // compute once, reuse below
        let process = SglangProcess {
            model_alias: manifest.alias.clone(),
            port,
            model_path,
            child,
            started_at: now,
            last_used: Arc::new(AtomicI64::new(now_ts)),
            logs,
        };

        self.processes.insert(manifest.alias.clone(), process);
        Ok(port)
    }

    /// Reuse an existing live process, or spawn one. Critical: all operations
    /// on the existing entry are SYNCHRONOUS (try_wait + atomic store) so we
    /// never hold the DashMap shard lock across an `.await` — that was a real
    /// hazard in earlier versions where slow reqs could deadlock the registry.
    ///
    /// Decision tree per alias:
    ///   * no entry → spawn fresh.
    ///   * entry alive (try_wait returned None) → touch last_used, return port.
    ///   * entry crashed → remove, then spawn fresh.
    pub async fn get_or_spawn(&self, alias: &str, manifest: &ModelManifest) -> Result<u16> {
        let result = match self.processes.get_mut(alias) {
            None => None,
            Some(mut entry) => match entry.child.try_wait() {
                Ok(None) => {
                    // Alive — touch timestamp lock-free, return port
                    entry
                        .last_used
                        .store(Utc::now().timestamp(), Ordering::Relaxed);
                    Some(Ok(entry.port))
                }
                Ok(Some(status)) => {
                    tracing::warn!(
                        "sglang process for {} exited with status {:?}, respawning",
                        alias,
                        status
                    );
                    Some(Err(()))
                }
                Err(e) => {
                    tracing::warn!("Error checking sglang process for {}: {}", alias, e);
                    Some(Err(()))
                }
            },
        };

        match result {
            Some(Ok(port)) => Ok(port),
            Some(Err(())) => {
                self.processes.remove(alias);
                self.spawn_coalesced(alias, manifest).await
            }
            None => self.spawn_coalesced(alias, manifest).await,
        }
    }

    /// A3 — Coalesce concurrent spawn requests for the same alias.
    /// First caller spawns; subsequent callers wait on a `Notify`. After the
    /// spawn finishes, all waiters re-check the `processes` map and reuse the
    /// fresh port. Saves duplicate Python/CUDA initialization under burst load.
    async fn spawn_coalesced(&self, alias: &str, manifest: &ModelManifest) -> Result<u16> {
        // Either become the spawner OR get a Notify to wait on.
        let (notify, am_spawner) = {
            let mut guard = self.spawn_in_flight.lock().await;
            if let Some(existing) = guard.get(alias) {
                (existing.clone(), false)
            } else {
                let n = Arc::new(tokio::sync::Notify::new());
                guard.insert(alias.to_string(), n.clone());
                (n, true)
            }
        };

        if am_spawner {
            // We're the lead spawner. Run the real spawn, then notify any
            // waiters and remove the in-flight marker — regardless of result.
            let result = self.spawn(manifest).await;
            {
                let mut guard = self.spawn_in_flight.lock().await;
                guard.remove(alias);
            }
            notify.notify_waiters();
            result
        } else {
            // Wait for the leader to finish, then look up the port.
            //
            // We do NOT rely solely on `notified()`: `Notify::notify_waiters()`
            // stores no permit, so a waiter that parks *after* the leader fires
            // would miss the wake and hang. Instead we re-check `processes` on a
            // short interval (the leader inserts the entry before notifying), so
            // correctness never depends on wake timing — at worst we poll a few
            // times. If the leader's spawn failed (marker gone, no entry), we
            // fall back to spawning ourselves.
            tracing::debug!("Coalescing sglang spawn for {} (waiting on leader)", alias);
            loop {
                let _ = tokio::time::timeout(Duration::from_millis(200), notify.notified()).await;
                if let Some(entry) = self.processes.get(alias) {
                    return Ok(entry.value().port);
                }
                // Leader done (marker cleared) but no process → its spawn failed.
                let leader_gone = {
                    let guard = self.spawn_in_flight.lock().await;
                    !guard.contains_key(alias)
                };
                if leader_gone {
                    return self.spawn(manifest).await;
                }
                // Otherwise the leader is still working — loop and wait again.
            }
        }
    }

    /// Remove from the map and kill the process tree. Waits up to 10s for
    /// the child to exit cleanly after the kill signal; logs a warning if it
    /// hangs but does not block indefinitely. Idempotent — calling on an
    /// already-dead alias is a no-op that returns `Ok(())`.
    pub async fn kill(&self, alias: &str) -> Result<()> {
        if let Some((_, mut process)) = self.processes.remove(alias) {
            tracing::info!("Killing sglang process tree for {}", alias);
            // Kill the whole process group/job, not just the direct child.
            kill_process_tree(&mut process.child).await;
            match tokio::time::timeout(Duration::from_secs(10), process.child.wait()).await {
                Ok(_) => {}
                Err(_) => tracing::warn!("sglang for {} did not exit within 10s", alias),
            }
        }
        Ok(())
    }

    /// Evict the least-recently-used sglang process. Used as a backstop when
    /// VRAM is full — a future caller can ask for a model and we free space by
    /// killing the staleyest one. Scan is entirely sync (atomic loads only).
    pub async fn evict_lru(&self) -> Result<()> {
        let mut oldest_alias: Option<String> = None;
        let mut oldest_ts: i64 = i64::MAX;

        for entry in self.processes.iter() {
            let ts = entry.value().last_used.load(Ordering::Relaxed);
            if ts < oldest_ts {
                oldest_ts = ts;
                oldest_alias = Some(entry.key().clone());
            }
        }

        if let Some(alias) = oldest_alias {
            tracing::info!("Evicting LRU sglang process: {}", alias);
            self.kill(&alias).await?;
        }

        Ok(())
    }

    /// Kill every process this manager owns. Called on daemon shutdown.
    /// Snapshots keys first so we don't hold a DashMap iterator across awaits.
    pub async fn kill_all(&self) -> Result<()> {
        let keys: Vec<String> = self.processes.iter().map(|e| e.key().clone()).collect();
        for alias in keys {
            self.kill(&alias).await?;
        }
        Ok(())
    }
}

/// Scan a port range and return the first one we can successfully `bind` to.
///
/// Caller must hold the manager's `port_alloc_lock` — the bind here is
/// dropped immediately (it was just a probe), so without the lock two
/// concurrent spawns could both probe the same port, both succeed, and then
/// both try to use it. The check+spawn window is still small but the lock
/// makes the common case race-free.
pub fn find_free_port(start: u16, end: u16) -> Result<u16> {
    for port in start..=end {
        if TcpListener::bind(format!("127.0.0.1:{}", port)).is_ok() {
            return Ok(port);
        }
    }
    Err(anyhow!("No free port in range {}..={}", start, end))
}

/// Place the child in its own process group (Unix) or process-group flag
/// (Windows). Without this, killing the direct child leaves grandchildren
/// orphaned — and sglang/Python forks workers, so that's the common case.
#[cfg(unix)]
pub fn configure_process_group(cmd: &mut Command) {
    cmd.process_group(0);
}

/// Windows variant of `configure_process_group`. The flag
/// `CREATE_NEW_PROCESS_GROUP` (0x200) lets `taskkill /T` target the whole tree.
#[cfg(windows)]
pub fn configure_process_group(cmd: &mut Command) {
    cmd.creation_flags(0x00000200);
}

/// Forcibly terminate `child` and its descendants.
///
///   * Unix: `kill -KILL -<pgid>` (negative PID = process group).
///   * Windows: `taskkill /F /T /PID <pid>`.
///
/// Then `child.kill().await` reaps the direct child either way.
pub async fn kill_process_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            // SIGKILL the entire process group (negative PID = group). The child
            // was spawned with process_group(0) so its PID == its PGID. We shell
            // out to `kill` to avoid pulling in a libc dependency. Ignored if the
            // process already exited.
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &format!("-{}", pid)])
                .status();
        }
    }
    #[cfg(windows)]
    {
        if let Some(pid) = child.id() {
            // taskkill /F (force) /T (tree). Synchronous; returns quickly.
            let _ = std::process::Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .output();
        }
    }
    // Always ensure the direct child is reaped.
    let _ = child.kill().await;
}

/// Send a tiny completion request to the freshly-spawned backend so it JIT-compiles
/// kernels, allocates KV cache pages, and loads the tokenizer. The first user
/// request then sees normal latency rather than 5–30 s cold-start cost.
/// Best-effort — backend-level errors are logged but never propagated.
pub async fn warmup_backend(client: &reqwest::Client, port: u16, alias: &str, backend: &str) {
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let body = serde_json::json!({
        "model": alias,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": false,
        "temperature": 0.0,
    });

    let started = std::time::Instant::now();
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!(
                "{} for {} warmed up in {}ms",
                backend,
                alias,
                started.elapsed().as_millis()
            );
        }
        Ok(resp) => {
            tracing::debug!(
                "{} warmup for {} returned status {} (non-fatal)",
                backend,
                alias,
                resp.status()
            );
        }
        Err(e) => {
            tracing::debug!("{} warmup for {} failed: {} (non-fatal)", backend, alias, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_free_port_returns_one_in_range() {
        // Pick a high port range very likely to be free in CI.
        let port = find_free_port(40000, 40100).expect("at least one free port");
        assert!((40000..=40100).contains(&port));
    }

    #[test]
    fn find_free_port_errors_when_range_invalid() {
        // start > end yields an empty range and an error.
        let result = find_free_port(50000, 49999);
        assert!(result.is_err());
    }

    #[test]
    fn find_free_port_skips_taken_port() {
        // Take a specific port, then make sure find_free_port doesn't return it.
        // Bind to an ephemeral port first so we know one definitely exists.
        let listener =
            TcpListener::bind("127.0.0.1:0").expect("can bind to ephemeral port");
        let taken = listener.local_addr().unwrap().port();

        // Tight range around the taken port. find_free_port should either skip
        // it (return a different port) or error if no other ports are available.
        let result = find_free_port(taken, taken);
        // The port we hold is taken, so binding to [taken, taken] fails.
        assert!(result.is_err());
        drop(listener);
    }
}
