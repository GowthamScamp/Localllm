//! # llama.cpp backend manager
//!
//! Mirror of `sglang.rs` for the llama-server binary. Used when sglang isn't
//! viable (no GPU, doesn't fit in VRAM, GGUF available). Reuses sglang's
//! port-finder, process-group setup, and tree-kill helpers — keeping the
//! cross-OS process plumbing in one place.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

use crate::config::{physical_cores, Settings};
use crate::gpu::VramManager;
use crate::inference::logbuf::LogRingBuffer;
use crate::inference::sglang::{configure_process_group, find_free_port, kill_process_tree};
use crate::registry::ModelManifest;

/// Upper bound on `--ctx-size`. The KV cache grows with context × parallel
/// slots, so uncapped 32k–128k contexts can exhaust RAM on CPU boxes for no
/// real benefit in typical chat. 8192 is plenty for interactive use; users who
/// need more can raise it once we expose a per-model override.
const MAX_CTX_SIZE: u32 = 8192;

/// One running llama-server process. See `SglangProcess` for the AtomicI64
/// last_used rationale — identical here.
pub struct LlamaCppServer {
    pub model_alias: String,
    pub port: u16,
    pub child: Child,
    pub started_at: DateTime<Utc>,
    pub last_used: Arc<AtomicI64>,
    /// Recent stdout/stderr lines (see SglangProcess::logs).
    pub logs: Arc<LogRingBuffer>,
}

/// Per-alias pool of llama-server processes. Same shape as `SglangManager`.
pub struct LlamaCppManager {
    pub processes: DashMap<String, LlamaCppServer>,
    pub settings: Arc<Settings>,
    port_alloc_lock: Mutex<()>,
    /// A3 — coalesces concurrent cold-spawn requests (see SglangManager).
    spawn_in_flight: Mutex<std::collections::HashMap<String, Arc<tokio::sync::Notify>>>,
}

impl LlamaCppManager {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self {
            processes: DashMap::new(),
            settings,
            port_alloc_lock: Mutex::new(()),
            spawn_in_flight: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Spawn a fresh llama-server. Requires:
    ///   * `manifest.gguf_path` to be set (model must be quantized first).
    ///   * `settings.llama_cpp_dir` to point at a built llama.cpp checkout
    ///     containing `build/bin/llama-server`.
    ///
    /// Adds `--embeddings` to the command when the manifest opts in — this
    /// flag enables `/v1/embeddings` on the server but disables generation.
    /// Health-poll loop matches `sglang.spawn` (100ms → 2s backoff).
    pub async fn spawn(&self, manifest: &ModelManifest) -> Result<u16> {
        let gguf_path = manifest
            .gguf_path
            .as_ref()
            .ok_or_else(|| {
                anyhow!(
                    "No GGUF path for '{}'. Run: localllm quantize {}",
                    manifest.alias,
                    manifest.alias
                )
            })?
            .clone();

        let llama_cpp_dir = self
            .settings
            .llama_cpp_dir
            .as_ref()
            .ok_or_else(|| anyhow!("llama_cpp_dir not configured"))?;

        // Cross-platform: resolves `llama-server` or `llama-server.exe` under
        // build/bin/, whichever exists on this OS.
        let server_bin = crate::platform::resolve_llama_binary(llama_cpp_dir, "llama-server")
            .ok_or_else(|| {
                anyhow!(
                    "llama-server not found under {:?}/build/bin/ (looked for both \
                     'llama-server' and 'llama-server.exe')",
                    llama_cpp_dir
                )
            })?;

        let port = {
            let _lock = self.port_alloc_lock.lock().await;
            find_free_port(
                self.settings.sglang_port_range_start,
                self.settings.sglang_port_range_end,
            )?
        };

        tracing::info!(
            "Spawning llama-server for {} on port {} with model {:?}",
            manifest.alias,
            port,
            gguf_path
        );

        // Hardware-aware, minimal configuration. Modern llama.cpp auto-fits the
        // model to whatever memory it finds, so we deliberately keep flags to a
        // minimum and let it make the GPU-vs-CPU and layer-offload decisions:
        //
        //   * GPU present → `--n-gpu-layers 999`: offload every layer that fits
        //     into VRAM (llama.cpp clamps to the real layer count and spills the
        //     remainder to CPU RAM automatically). This is the big speed lever.
        //   * No GPU      → omit the flag entirely → pure CPU, using system RAM.
        //
        // We don't micromanage --mlock / --batch-size / --n-predict anymore;
        // their defaults are good and the manual values caused trouble on
        // low-RAM and newer-build machines.
        let has_gpu = !VramManager::query_gpus().unwrap_or_default().is_empty();

        // Context size: cap at a sane ceiling so the KV cache (which scales with
        // ctx × parallel slots) doesn't blow out RAM on CPU boxes. Most chat use
        // never needs the model's full trained context (often 32k–128k).
        let ctx = manifest.context_length.clamp(2048, MAX_CTX_SIZE);

        // Threads. Decode is latency-bound and best on physical cores (HT
        // siblings contend); prefill is compute-bound and scales with ALL logical
        // cores. So: --threads = physical cores (or config), --threads-batch =
        // logical cores. This noticeably speeds prompt processing on CPU.
        let threads = self.settings.llamacpp_threads.unwrap_or_else(physical_cores);
        let threads_batch = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(threads);

        let mut cmd = Command::new(&server_bin);
        cmd.arg("--model")
            .arg(&gguf_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(ctx.to_string())
            // Parallel decode slots: serve a few concurrent requests without
            // serializing. Kept modest so KV-cache memory stays reasonable.
            .arg("-np")
            .arg(self.settings.llamacpp_parallel_slots.to_string())
            .arg("--threads")
            .arg(threads.to_string())
            .arg("--threads-batch")
            .arg(threads_batch.to_string())
            // FlashAttention: `on` when enabled in config, else `auto` (let
            // llama.cpp decide; never force-disable a beneficial default). The
            // explicit value form is required by recent builds — a bare
            // --flash-attn would swallow the next argument.
            .arg("--flash-attn")
            .arg(if self.settings.llamacpp_flash_attn { "on" } else { "auto" });

        // Batch sizes: only override when the user changed the default (512). The
        // server's own defaults (batch 2048 / ubatch 512) are well-tuned; passing
        // our 512 default would *reduce* prefill throughput, so we leave it alone
        // unless explicitly configured to something else.
        const DEFAULT_BATCH: u32 = 512;
        if self.settings.llamacpp_batch_size != DEFAULT_BATCH {
            cmd.arg("--batch-size")
                .arg(self.settings.llamacpp_batch_size.to_string())
                .arg("--ubatch-size")
                .arg(self.settings.llamacpp_batch_size.min(512).to_string());
        }

        // Lock weights in RAM to avoid mid-generation paging stalls. mlock
        // failure (low RLIMIT_MEMLOCK, common in containers/WSL) is non-fatal in
        // llama-server — it just warns and continues.
        if self.settings.llamacpp_mlock {
            cmd.arg("--mlock");
        }

        if has_gpu {
            tracing::info!("llama-server {}: GPU detected, offloading layers to VRAM", manifest.alias);
            cmd.arg("--n-gpu-layers").arg("999");
        } else {
            tracing::info!("llama-server {}: no GPU, running on CPU (system RAM)", manifest.alias);
        }

        // Enable the /v1/embeddings endpoint when the manifest opts in.
        if manifest.embeddings {
            cmd.arg("--embeddings");
        }

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // A8 — process group / job object for clean kill of any grandchildren.
        configure_process_group(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn llama-server: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture llama-server stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("Failed to capture llama-server stderr"))?;

        let alias_stdout = manifest.alias.clone();
        let alias_stderr = manifest.alias.clone();

        let logs = Arc::new(LogRingBuffer::default());
        let logs_stdout = Arc::clone(&logs);
        let logs_stderr = Arc::clone(&logs);

        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::debug!("[llamacpp:{}:stdout] {}", alias_stdout, line);
                logs_stdout.push(format!("[stdout] {}", line));
            }
        });

        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::debug!("[llamacpp:{}:stderr] {}", alias_stderr, line);
                logs_stderr.push(format!("[stderr] {}", line));
            }
        });

        // Health-check polling with exponential backoff
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
                        "llama-server for {} ready on port {} after {}ms",
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
                    "llama-server startup timeout after {}s for {}",
                    self.settings.sglang_startup_timeout_secs,
                    manifest.alias
                ));
            }

            delay_ms = (delay_ms.saturating_mul(2)).min(2000);
        }

        // No explicit HTTP warm-up here: llama-server already performs an internal
        // warm-up run (`--warmup`, on by default) *during* startup, before its
        // /health returns 200. By the time the poll above breaks out, the model is
        // already warm — an extra warm-up request would only add latency to the
        // first-ready path. (sglang still warms up explicitly; it has no built-in.)

        let now = Utc::now();
        let now_ts = now.timestamp(); // compute once, reuse below
        let server = LlamaCppServer {
            model_alias: manifest.alias.clone(),
            port,
            child,
            started_at: now,
            last_used: Arc::new(AtomicI64::new(now_ts)),
            logs,
        };

        self.processes.insert(manifest.alias.clone(), server);
        Ok(port)
    }

    /// Reuse a live llama-server or spawn one. Same sync-only existing-entry
    /// pattern as `SglangManager::get_or_spawn` — never holds the shard lock
    /// across an await.
    pub async fn get_or_spawn(&self, alias: &str, manifest: &ModelManifest) -> Result<u16> {
        let result = match self.processes.get_mut(alias) {
            None => None,
            Some(mut entry) => match entry.child.try_wait() {
                Ok(None) => {
                    entry
                        .last_used
                        .store(Utc::now().timestamp(), Ordering::Relaxed);
                    Some(Ok(entry.port))
                }
                Ok(Some(status)) => {
                    tracing::warn!(
                        "llama-server for {} exited with status {:?}, respawning",
                        alias,
                        status
                    );
                    Some(Err(()))
                }
                Err(e) => {
                    tracing::warn!("Error checking llama-server for {}: {}", alias, e);
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

    /// A3 — Coalesce concurrent spawn requests (see SglangManager::spawn_coalesced).
    async fn spawn_coalesced(&self, alias: &str, manifest: &ModelManifest) -> Result<u16> {
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
            let result = self.spawn(manifest).await;
            {
                let mut guard = self.spawn_in_flight.lock().await;
                guard.remove(alias);
            }
            notify.notify_waiters();
            result
        } else {
            // Re-check on a short interval rather than depending on the wake.
            // `notify_waiters()` stores no permit, so a waiter that parks after
            // the leader fires would hang; polling `processes` makes correctness
            // independent of wake timing. See SglangManager::spawn_coalesced.
            tracing::debug!("Coalescing llama-server spawn for {}", alias);
            loop {
                let _ = tokio::time::timeout(Duration::from_millis(200), notify.notified()).await;
                if let Some(entry) = self.processes.get(alias) {
                    return Ok(entry.value().port);
                }
                let leader_gone = {
                    let guard = self.spawn_in_flight.lock().await;
                    !guard.contains_key(alias)
                };
                if leader_gone {
                    return self.spawn(manifest).await;
                }
            }
        }
    }

    /// Kill a specific llama-server. Idempotent; 10s grace period.
    pub async fn kill(&self, alias: &str) -> Result<()> {
        if let Some((_, mut server)) = self.processes.remove(alias) {
            tracing::info!("Killing llama-server process tree for {}", alias);
            kill_process_tree(&mut server.child).await;
            match tokio::time::timeout(Duration::from_secs(10), server.child.wait()).await {
                Ok(_) => {}
                Err(_) => tracing::warn!("llama-server for {} did not exit within 10s", alias),
            }
        }
        Ok(())
    }

    /// Evict the least-recently-used llama-server. See `SglangManager::evict_lru`.
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
            tracing::info!("Evicting LRU llama-server process: {}", alias);
            self.kill(&alias).await?;
        }

        Ok(())
    }

    /// Kill every llama-server this manager owns. Called on daemon shutdown.
    pub async fn kill_all(&self) -> Result<()> {
        let keys: Vec<String> = self.processes.iter().map(|e| e.key().clone()).collect();
        for alias in keys {
            self.kill(&alias).await?;
        }
        Ok(())
    }
}
