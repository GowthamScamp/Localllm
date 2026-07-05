//! # Inference backend selection
//!
//! Two backends are managed by sibling modules:
//!   * `sglang` — Python-based, GPU-only, continuous batching + RadixAttention.
//!     Best throughput at scale but needs CUDA and a fitting model.
//!   * `llamacpp` — C++-based, runs anywhere, supports quantized GGUF.
//!     Best for CPU or VRAM-constrained GPU setups.
//!
//! `InferenceRouter` picks one per-request based on VRAM availability and
//! whether a GGUF file exists. The routing is deterministic; once a process
//! is spawned for a given alias, every subsequent request reuses it.

pub mod llamacpp;
pub mod logbuf;
pub mod sglang;

pub use llamacpp::LlamaCppManager;
pub use sglang::SglangManager;

use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::config::Settings;
use crate::gpu::VramManager;
use crate::registry::ModelManifest;

/// Routes each request to either sglang or llama.cpp based on hardware fit.
/// Holds Arcs to both managers; cheap to clone, used inside the router.
pub struct InferenceRouter {
    pub sglang_manager: Arc<SglangManager>,
    pub llamacpp_manager: Arc<LlamaCppManager>,
    pub settings: Arc<Settings>,
}

impl InferenceRouter {
    pub fn new(
        sglang_manager: Arc<SglangManager>,
        llamacpp_manager: Arc<LlamaCppManager>,
        settings: Arc<Settings>,
    ) -> Self {
        Self {
            sglang_manager,
            llamacpp_manager,
            settings,
        }
    }

    /// Return the local HTTP endpoint serving this model, spawning the backend
    /// if needed. Routing decision (in order):
    ///   1. GPU available AND full-precision weights fit in VRAM → sglang.
    ///   2. Otherwise, if a GGUF file exists → llama.cpp.
    ///   3. Otherwise, error with a hint to run `localllm quantize`.
    ///
    /// Once a backend is spawned for an alias, it stays alive until either the
    /// TTL eviction loop kills it or a kill_all on shutdown takes it down.
    pub async fn get_endpoint(&self, manifest: &ModelManifest) -> Result<String> {
        let alias = &manifest.alias;

        // Cached + non-blocking: avoids spawning nvidia-smi on every request and
        // never stalls the tokio worker on the (rare) cache-miss probe.
        let gpus = VramManager::query_gpus_async().await?;

        if !gpus.is_empty() && VramManager::can_fit(manifest, &gpus) {
            tracing::info!("Using sglang for {} (GPU fit confirmed)", alias);
            let port = self
                .sglang_manager
                .get_or_spawn(alias, manifest)
                .await?;
            return Ok(format!("http://127.0.0.1:{}", port));
        }

        if manifest.gguf_path.is_some() {
            tracing::info!("Using llama.cpp for {} (no GPU or doesn't fit)", alias);
            let port = self
                .llamacpp_manager
                .get_or_spawn(alias, manifest)
                .await?;
            return Ok(format!("http://127.0.0.1:{}", port));
        }

        // No GPU fit and no GGUF on disk. The model can't run as-is: llama.cpp
        // needs a GGUF, and sglang needs a fitting GPU. Give an actionable hint
        // instead of a dead end — re-pulling fetches a runnable prebuilt GGUF on
        // CPU boxes (the common case), and quantize covers the torch path.
        Err(anyhow!(
            "No runnable backend for '{alias}': it has no GGUF and doesn't fit in GPU VRAM. \
             Re-pull to fetch a prebuilt GGUF (localllm pull <repo>), or convert locally \
             if you have Python+torch (localllm quantize {alias}).",
            alias = alias
        ))
    }

    /// Kill every backend process in both pools. Called on daemon shutdown
    /// so we don't leak Python/sglang processes after a SIGTERM.
    /// Both kills are awaited; the second error doesn't mask the first.
    pub async fn kill_all(&self) -> Result<()> {
        let sg = self.sglang_manager.kill_all().await;
        let lc = self.llamacpp_manager.kill_all().await;
        sg?;
        lc?;
        Ok(())
    }
}
