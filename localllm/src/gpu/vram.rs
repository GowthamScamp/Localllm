//! # GPU detection and VRAM fitting
//!
//! Wraps `nvidia-smi` to enumerate GPUs and estimate whether a model fits.
//! NVIDIA-only for now; ROCm/Metal would need new probe functions.
//!
//! The fit check is a heuristic (parameter count × bytes/param + KV cache,
//! plus 15% headroom) — not exact. If it's wrong, the downstream sglang spawn
//! will OOM and surface a clear error; this just keeps us from trying obvious
//! losers.

use anyhow::Result;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::registry::ModelManifest;

/// One GPU's identity and live memory state. `free_vram_mb` updates between
/// `query_gpus()` calls — this is a snapshot, not a subscription.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub total_vram_mb: u64,
    pub free_vram_mb: u64,
}

/// How long a `query_gpus()` result is reused before re-probing `nvidia-smi`.
/// GPU presence is effectively static; free-VRAM drifts slowly. 3s dedupes a
/// burst of requests while staying fresh enough for routing decisions.
const GPU_CACHE_TTL: Duration = Duration::from_secs(3);

/// Process-wide cache of the last `nvidia-smi` probe. `None` until first probe.
/// Guarded by a std Mutex — held only for the microseconds it takes to read or
/// swap the Vec, never across the subprocess call.
static GPU_CACHE: Mutex<Option<(Instant, Vec<GpuInfo>)>> = Mutex::new(None);

/// Stateless namespace for GPU probing. Methods are all static; no instance
/// needs to be constructed. Kept as a struct (not a free-function module) for
/// future expansion (e.g. `VramManager::with_backend(GpuBackend::Rocm)`).
pub struct VramManager;

impl VramManager {
    /// Cached GPU enumeration. Returns the last probe if it's younger than
    /// `GPU_CACHE_TTL`, otherwise re-probes `nvidia-smi`. This is the function
    /// hot paths should call — it avoids spawning a subprocess on every request.
    ///
    /// Still synchronous: callers inside async contexts should use
    /// [`query_gpus_async`] so the (rare) cache-miss subprocess spawn doesn't
    /// block a tokio worker thread.
    pub fn query_gpus() -> Result<Vec<GpuInfo>> {
        // Fast path: fresh cache.
        if let Ok(guard) = GPU_CACHE.lock() {
            if let Some((at, gpus)) = guard.as_ref() {
                if at.elapsed() < GPU_CACHE_TTL {
                    return Ok(gpus.clone());
                }
            }
        }

        // Slow path: re-probe and refresh the cache.
        let gpus = Self::probe_gpus()?;
        if let Ok(mut guard) = GPU_CACHE.lock() {
            *guard = Some((Instant::now(), gpus.clone()));
        }
        Ok(gpus)
    }

    /// Async, non-blocking GPU query. On a cache hit returns immediately; on a
    /// miss it runs the `nvidia-smi` probe on a blocking thread pool so the
    /// tokio worker isn't stalled by the subprocess spawn.
    pub async fn query_gpus_async() -> Result<Vec<GpuInfo>> {
        // Cache hit → no thread hop needed.
        if let Ok(guard) = GPU_CACHE.lock() {
            if let Some((at, gpus)) = guard.as_ref() {
                if at.elapsed() < GPU_CACHE_TTL {
                    return Ok(gpus.clone());
                }
            }
        }
        // Miss → probe off the async runtime.
        tokio::task::spawn_blocking(Self::query_gpus)
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("GPU probe task failed: {}", e)))
    }

    /// Raw, uncached `nvidia-smi` probe. Prefer [`query_gpus`] /
    /// [`query_gpus_async`] which cache the result.
    ///
    /// Returns `Ok(vec![])` (not Err) when:
    ///   * `nvidia-smi` is missing (no NVIDIA driver installed).
    ///   * `nvidia-smi` runs but exits non-zero (e.g. driver loaded but no GPU).
    ///
    /// This means callers can treat "no GPU" and "no NVIDIA tools" identically —
    /// the routing decision becomes "fall back to llama.cpp" either way.
    /// Genuinely surprising failures (e.g. one parse error per line) get logged
    /// at warn level but don't fail the call.
    pub fn probe_gpus() -> Result<Vec<GpuInfo>> {
        let output = match std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    tracing::debug!("nvidia-smi not found, no GPU info available");
                    return Ok(vec![]);
                }
                tracing::warn!("nvidia-smi failed: {}", e);
                return Ok(vec![]);
            }
        };

        if !output.status.success() {
            tracing::warn!("nvidia-smi exited with non-zero status");
            return Ok(vec![]);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();

        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(4, ',').collect();
            if parts.len() < 4 {
                tracing::warn!("Unexpected nvidia-smi output line: {}", line);
                continue;
            }
            let index = match parts[0].trim().parse::<u32>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let name = parts[1].trim().to_string();
            let total_vram_mb = match parts[2].trim().parse::<u64>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let free_vram_mb = match parts[3].trim().parse::<u64>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            gpus.push(GpuInfo {
                index,
                name,
                total_vram_mb,
                free_vram_mb,
            });
        }

        Ok(gpus)
    }

    /// Rough VRAM requirement in MiB. Sum of:
    ///   * Weight memory = params × bytes_per_param (bfloat16 default).
    ///   * KV cache estimate = `context_length × 64 / 1024` MiB (a crude
    ///     approximation; real cache depends on head count, head dim, dtype,
    ///     and batch size).
    ///
    /// Underestimates for big-context models; overestimates for low-batch use.
    /// Good enough for go/no-go routing.
    pub fn estimate_model_vram_mb(manifest: &ModelManifest) -> u64 {
        let bytes_per_param = manifest
            .quantization
            .as_ref()
            .map(|q| q.bytes_per_param())
            .unwrap_or(2.0_f32);

        let weight_mb =
            (manifest.parameters_billion * 1e9 * bytes_per_param / 1024.0 / 1024.0) as u64;
        let kv_cache_mb = manifest.context_length as u64 * 64 / 1024;

        weight_mb + kv_cache_mb
    }

    /// Decide whether `manifest` can plausibly fit across the available GPUs.
    /// Sums free VRAM across all reported GPUs (assumes multi-GPU sharding
    /// works, which sglang supports natively). Adds 15% headroom over the
    /// raw estimate to absorb estimation error.
    pub fn can_fit(manifest: &ModelManifest, gpus: &[GpuInfo]) -> bool {
        let total_free: u64 = gpus.iter().map(|g| g.free_vram_mb).sum();
        let required = Self::estimate_model_vram_mb(manifest) * 115 / 100;
        total_free >= required
    }

    /// Heuristic transformer layer count from parameter scale. Real architectures
    /// vary, but this tracks the common families closely enough for offload
    /// decisions (we only need a ballpark to size the GPU split):
    ///   ≤2B → 24, ≤4B → 32, ≤9B → 32, ≤15B → 40, ≤35B → 60, ≤80B → 80, else 96.
    pub fn estimate_layer_count(parameters_billion: f32) -> u32 {
        match parameters_billion {
            p if p <= 2.0 => 24,
            p if p <= 4.0 => 32,
            p if p <= 9.0 => 32,
            p if p <= 15.0 => 40,
            p if p <= 35.0 => 60,
            p if p <= 80.0 => 80,
            _ => 96,
        }
    }

    /// Recommend a value for llama.cpp's `--n-gpu-layers`.
    ///   * No GPUs → `0` (pure CPU; the caller omits the flag).
    ///   * Whole model fits in free VRAM → `999` (llama.cpp clamps to actual
    ///     layer count, offloading everything).
    ///   * Partial fit → offload `floor(free * 0.9 / per_layer_mb)` layers so
    ///     hot layers run on GPU and the remainder spills to CPU.
    ///
    /// `0.9` headroom leaves room for the KV cache and CUDA context.
    pub fn recommend_gpu_layers(manifest: &ModelManifest, gpus: &[GpuInfo]) -> u32 {
        if gpus.is_empty() {
            return 0;
        }
        if Self::can_fit(manifest, gpus) {
            return 999; // offload all; llama.cpp clamps to the real layer count
        }

        let total_free: u64 = gpus.iter().map(|g| g.free_vram_mb).sum();
        let layers = Self::estimate_layer_count(manifest.parameters_billion).max(1);
        // Weight-only per-layer estimate (exclude the KV-cache term so we don't
        // double-penalize; KV cache is what the 0.9 headroom covers).
        let bytes_per_param = manifest
            .quantization
            .as_ref()
            .map(|q| q.bytes_per_param())
            .unwrap_or(2.0_f32);
        let weight_mb =
            (manifest.parameters_billion * 1e9 * bytes_per_param / 1024.0 / 1024.0) as u64;
        let per_layer_mb = (weight_mb / layers as u64).max(1);

        let budget = (total_free as f64 * 0.9) as u64;
        let fit_layers = (budget / per_layer_mb) as u32;
        fit_layers.min(layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::manifest::{ModelManifest, WeightFormat};

    fn manifest(params_b: f32) -> ModelManifest {
        let now = chrono::Utc::now();
        ModelManifest {
            repo_id: "test/model".into(),
            alias: "test".into(),
            revision: "main".into(),
            local_path: std::path::PathBuf::from("/tmp/test"),
            architecture: "llama".into(),
            weight_format: WeightFormat::GGUF,
            parameters_billion: params_b,
            context_length: 4096,
            quantization: None,
            gguf_path: None,
            files: vec![],
            downloaded_at: now,
            last_used: now,
            embeddings: false,
            modelfile: None,
        }
    }

    fn gpu(free_mb: u64) -> GpuInfo {
        GpuInfo {
            index: 0,
            name: "Test GPU".into(),
            total_vram_mb: free_mb,
            free_vram_mb: free_mb,
        }
    }

    #[test]
    fn recommend_gpu_layers_zero_without_gpu() {
        // No GPUs → always CPU-only (0 layers).
        assert_eq!(VramManager::recommend_gpu_layers(&manifest(7.0), &[]), 0);
    }

    #[test]
    fn recommend_gpu_layers_all_when_fits() {
        // 7B model, 24 GiB free → fits easily → offload everything (999, clamped
        // by llama.cpp downstream).
        let layers = VramManager::recommend_gpu_layers(&manifest(7.0), &[gpu(24_000)]);
        assert_eq!(layers, 999);
    }

    #[test]
    fn recommend_gpu_layers_partial_when_tight() {
        // 70B model at bf16 needs ~134 GiB; only 16 GiB free → partial offload,
        // strictly fewer than the full layer count and > 0.
        let m = manifest(70.0);
        let full = VramManager::estimate_layer_count(m.parameters_billion);
        let layers = VramManager::recommend_gpu_layers(&m, &[gpu(16_000)]);
        assert!(layers < full, "expected partial offload, got {}", layers);
    }

    #[test]
    fn estimate_layer_count_scales_with_size() {
        assert!(
            VramManager::estimate_layer_count(1.0) <= VramManager::estimate_layer_count(70.0)
        );
        assert_eq!(VramManager::estimate_layer_count(1.5), 24);
    }

    #[test]
    fn physical_cores_at_least_one() {
        assert!(crate::config::physical_cores() >= 1);
    }

    #[test]
    fn query_gpus_is_cached() {
        // First call populates the cache; a second call within the TTL must
        // return the same data without re-probing. On a box with no nvidia-smi
        // both return an empty Vec — the point is that it doesn't error and the
        // cache machinery works. (We can't assert subprocess-spawn count here,
        // but we can assert idempotent, fast, error-free repeat calls.)
        let first = VramManager::query_gpus().expect("query ok");
        let second = VramManager::query_gpus().expect("cached query ok");
        assert_eq!(first.len(), second.len());
    }

    #[tokio::test]
    async fn query_gpus_async_matches_sync() {
        let sync = VramManager::query_gpus().expect("sync ok");
        let asyncd = VramManager::query_gpus_async().await.expect("async ok");
        assert_eq!(sync.len(), asyncd.len());
    }
}
