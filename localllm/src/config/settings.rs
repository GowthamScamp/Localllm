//! # Settings
//!
//! Process-wide configuration loaded from (in precedence order):
//!   1. Environment variables — `HF_TOKEN`, `LOCALLLM_DAEMON_HOST`, etc.
//!   2. `~/.localllm/config.toml` — optional, all keys optional.
//!   3. Built-in defaults defined by the `default_*` functions below.
//!
//! Every field has a `serde(default = "...")` so partial config files work —
//! omit a key to inherit the default. Adding a new tunable means: add the field,
//! add a `default_<name>()` function, wire it into `Default::default()`, and
//! optionally add an env-var override in `Settings::load`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Read `HF_TOKEN` from the environment, treating empty / whitespace-only
/// values as absent. A blank token would otherwise be sent as
/// `Authorization: Bearer ` and rejected by HuggingFace with HTTP 401 — worse
/// than sending no auth at all (public models would stop downloading).
fn hf_token_from_env() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// All runtime-configurable knobs for the daemon and CLI.
///
/// Shared across async tasks via `Arc<Settings>` — never mutated after `load()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_daemon_host")]
    pub daemon_host: String,

    #[serde(default = "default_daemon_port")]
    pub daemon_port: u16,

    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    #[serde(default = "default_gguf_dir")]
    pub gguf_dir: PathBuf,

    #[serde(default = "default_manifests_dir")]
    pub manifests_dir: PathBuf,

    #[serde(skip)]
    pub hf_token: Option<String>,

    #[serde(default = "default_sglang_port_range_start")]
    pub sglang_port_range_start: u16,

    #[serde(default = "default_sglang_port_range_end")]
    pub sglang_port_range_end: u16,

    #[serde(default = "default_max_concurrent_downloads")]
    pub max_concurrent_downloads: usize,

    #[serde(default = "default_download_chunk_size_bytes")]
    pub download_chunk_size_bytes: usize,

    #[serde(default = "default_sglang_startup_timeout_secs")]
    pub sglang_startup_timeout_secs: u64,

    #[serde(default)]
    pub llama_cpp_dir: Option<PathBuf>,

    #[serde(default = "default_model_ttl_secs")]
    pub model_ttl_secs: u64,

    /// List of model aliases to pre-warm on daemon startup. The daemon spawns
    /// each in a background task immediately after the manifest cache is loaded,
    /// so first-request latency for these models is near zero.
    ///
    /// Example: `preload = ["llama-3.2-1b-instruct", "tinyllama-1.1b-chat-v1.0"]`
    #[serde(default)]
    pub preload: Vec<String>,

    /// How many parallel decode slots to allocate inside `llama-server` via
    /// `-np`. Each slot can handle one in-flight request without serializing.
    /// Memory cost is small (mostly KV cache slots); throughput cost is large
    /// when slots == 1 and you have N>1 simultaneous chat clients.
    #[serde(default = "default_llamacpp_parallel_slots")]
    pub llamacpp_parallel_slots: u32,

    /// Enable FlashAttention in llama-server (`--flash-attn`). Faster prompt
    /// processing and smaller KV cache. On by default — modern llama.cpp builds
    /// support it broadly. Set false if your build predates FA support.
    #[serde(default = "default_true")]
    pub llamacpp_flash_attn: bool,

    /// CPU decode threads for llama-server (`--threads`). `None` = auto-detect
    /// physical core count (best for dedicated boxes). `Some(n)` pins it — use
    /// a lower value when sharing the machine with other CPU-heavy work.
    #[serde(default)]
    pub llamacpp_threads: Option<usize>,

    /// Lock model weights into RAM (`--mlock`) so the OS can't page them out
    /// mid-generation. Avoids latency spikes under memory pressure. On by default.
    #[serde(default = "default_true")]
    pub llamacpp_mlock: bool,

    /// Logical batch size for llama-server (`--batch-size`). Larger values
    /// improve prompt-processing throughput at some memory cost. 512 is a
    /// good balance for most consumer GPUs/CPUs.
    #[serde(default = "default_llamacpp_batch_size")]
    pub llamacpp_batch_size: u32,

    /// sglang prefill chunk size (`--chunked-prefill-size`). Chunking long
    /// prompts keeps the scheduler responsive and improves throughput on
    /// large-context requests.
    #[serde(default = "default_sglang_chunked_prefill_size")]
    pub sglang_chunked_prefill_size: u32,

    /// Enable `--enable-torch-compile` in sglang. Speeds steady-state decoding
    /// but adds minutes to the *first* spawn while Torch compiles kernels.
    /// Off by default so spawns stay fast; turn on for long-lived servers.
    #[serde(default)]
    pub sglang_torch_compile: bool,

    /// sglang static memory fraction (`--mem-fraction-static`). Fraction of GPU
    /// VRAM sglang reserves up-front. Lower it (e.g. 0.7) if you share the GPU
    /// with other processes.
    #[serde(default = "default_sglang_mem_fraction")]
    pub sglang_mem_fraction: f32,
}

fn default_llamacpp_parallel_slots() -> u32 {
    // 2 concurrent decode slots — enough for interactive use without the KV
    // cache (which scales with slots × ctx) eating too much RAM on small boxes.
    2
}

fn default_true() -> bool {
    true
}

fn default_llamacpp_batch_size() -> u32 {
    512
}

fn default_sglang_chunked_prefill_size() -> u32 {
    8192
}

fn default_sglang_mem_fraction() -> f32 {
    0.85
}

/// Estimate physical CPU core count. `available_parallelism()` reports logical
/// cores (includes hyperthreads); halving approximates physical cores, which is
/// the sweet spot for llama.cpp decode threads (HT siblings contend on the same
/// execution units). Floored at 1.
pub fn physical_cores() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    (logical / 2).max(1)
}

/// Daemon binds to loopback by default — local-only, no exposure to LAN.
/// Override with `LOCALLLM_DAEMON_HOST=0.0.0.0` to expose deliberately.
fn default_daemon_host() -> String {
    "127.0.0.1".to_string()
}

/// Default port, picked to avoid collision with Ollama's 11434.
fn default_daemon_port() -> u16 {
    11435
}

/// `~/.localllm/models` — original HuggingFace safetensors live here.
fn default_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".localllm")
        .join("models")
}

/// `~/.localllm/gguf` — quantized GGUF outputs from llama-quantize live here.
fn default_gguf_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".localllm")
        .join("gguf")
}

/// `~/.localllm/manifests` — one `<alias>.json` file per registered model.
fn default_manifests_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".localllm")
        .join("manifests")
}

/// Lower bound of the port range used by both sglang AND llama-server inference
/// processes. Picked to be well above the IANA ephemeral range.
fn default_sglang_port_range_start() -> u16 {
    30000
}

/// Upper bound of the inference-process port range (inclusive).
fn default_sglang_port_range_end() -> u16 {
    31000
}

/// How many HF files to download in parallel during a single `pull`. Higher
/// values saturate bandwidth faster but risk HF rate-limiting on large repos.
fn default_max_concurrent_downloads() -> usize {
    4
}

/// Default streaming chunk size (64 MiB). Reqwest streams in smaller increments
/// anyway; this is only a soft hint.
fn default_download_chunk_size_bytes() -> usize {
    67108864
}

/// How long to wait for an sglang or llama-server `/health` to respond after
/// spawn before giving up and killing the child. Large CUDA models can take
/// 30-60s to warm up; default is generous.
fn default_sglang_startup_timeout_secs() -> u64 {
    120
}

/// Idle TTL — an inference process not used for this many seconds is killed by
/// the background eviction loop. 5 minutes by default; bump for dev / debug.
fn default_model_ttl_secs() -> u64 {
    300
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            daemon_host: default_daemon_host(),
            daemon_port: default_daemon_port(),
            models_dir: default_models_dir(),
            gguf_dir: default_gguf_dir(),
            manifests_dir: default_manifests_dir(),
            hf_token: hf_token_from_env(),
            sglang_port_range_start: default_sglang_port_range_start(),
            sglang_port_range_end: default_sglang_port_range_end(),
            max_concurrent_downloads: default_max_concurrent_downloads(),
            download_chunk_size_bytes: default_download_chunk_size_bytes(),
            sglang_startup_timeout_secs: default_sglang_startup_timeout_secs(),
            llama_cpp_dir: None,
            model_ttl_secs: default_model_ttl_secs(),
            preload: Vec::new(),
            llamacpp_parallel_slots: default_llamacpp_parallel_slots(),
            llamacpp_flash_attn: default_true(),
            llamacpp_threads: None,
            llamacpp_mlock: default_true(),
            llamacpp_batch_size: default_llamacpp_batch_size(),
            sglang_chunked_prefill_size: default_sglang_chunked_prefill_size(),
            sglang_torch_compile: false,
            sglang_mem_fraction: default_sglang_mem_fraction(),
        }
    }
}

impl Settings {
    /// Load settings from disk + environment.
    ///
    /// Reads `~/.localllm/config.toml` if present (missing file is fine —
    /// returns `Settings::default()`). Then overlays env-var overrides on top,
    /// so env always wins over file. Parse errors on the TOML are fatal —
    /// silently falling back would mask user mistakes.
    pub fn load() -> Result<Self> {
        let config_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".localllm")
            .join("config.toml");

        let mut settings: Settings = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            toml::from_str(&content).map_err(|e| anyhow::anyhow!("Config parse error: {}", e))?
        } else {
            Settings::default()
        };

        // Override with environment variables
        settings.hf_token = hf_token_from_env();

        if let Ok(host) = std::env::var("LOCALLLM_DAEMON_HOST") {
            settings.daemon_host = host;
        }
        if let Ok(port_str) = std::env::var("LOCALLLM_DAEMON_PORT") {
            if let Ok(port) = port_str.parse::<u16>() {
                settings.daemon_port = port;
            }
        }
        if let Ok(dir) = std::env::var("LOCALLLM_MODELS_DIR") {
            settings.models_dir = PathBuf::from(dir);
        }
        if let Ok(dir) = std::env::var("LOCALLLM_GGUF_DIR") {
            settings.gguf_dir = PathBuf::from(dir);
        }
        if let Ok(dir) = std::env::var("LOCALLLM_MANIFESTS_DIR") {
            settings.manifests_dir = PathBuf::from(dir);
        }
        if let Ok(dir) = std::env::var("LOCALLLM_LLAMA_CPP_DIR") {
            settings.llama_cpp_dir = Some(PathBuf::from(dir));
        }

        // B3 — auto-detect llama_cpp_dir if not configured. We look in a short
        // list of common install locations. Removes the "Compression engine
        // unavailable" footgun when the user has llama.cpp built somewhere standard.
        if settings.llama_cpp_dir.is_none() {
            settings.llama_cpp_dir = autodetect_llama_cpp_dir();
            if let Some(ref dir) = settings.llama_cpp_dir {
                tracing::info!("Auto-detected llama_cpp_dir at {:?}", dir);
            }
        }

        Ok(settings)
    }

    /// Create `models_dir`, `gguf_dir`, and `manifests_dir` if they don't exist.
    /// Called once at daemon startup so later code can assume the paths are usable.
    pub fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(&self.gguf_dir)?;
        std::fs::create_dir_all(&self.manifests_dir)?;
        Ok(())
    }

    /// Compose the daemon's HTTP base URL from `daemon_host` + `daemon_port`.
    /// Always `http://` — TLS is intentionally not supported for a local daemon.
    pub fn daemon_url(&self) -> String {
        format!("http://{}:{}", self.daemon_host, self.daemon_port)
    }
}

/// Look for `build/bin/llama-server` (or `.exe` on Windows) in a short list of
/// common locations. Returns the parent directory of the build/ subtree, which
/// is the format `llama_cpp_dir` expects.
fn autodetect_llama_cpp_dir() -> Option<PathBuf> {
    let home = dirs::home_dir();
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(ref h) = home {
        // The managed location where first-run setup clones + builds llama.cpp
        // (~/.localllm/llama.cpp). Checked first so a setup-built engine is
        // picked up even before the daemon's own ensure_llama_cpp runs.
        candidates.push(h.join(".localllm").join("llama.cpp"));
        candidates.push(h.join("llama.cpp"));
        candidates.push(h.join("src").join("llama.cpp"));
        candidates.push(h.join("code").join("llama.cpp"));
        candidates.push(h.join(".local").join("llama.cpp"));
    }
    if cfg!(windows) {
        candidates.push(PathBuf::from("C:/llama.cpp"));
        candidates.push(PathBuf::from("C:/Program Files/llama.cpp"));
    } else {
        candidates.push(PathBuf::from("/usr/local/llama.cpp"));
        candidates.push(PathBuf::from("/opt/llama.cpp"));
    }

    // Use the shared resolver so `.exe`/bare-name handling stays in one place.
    candidates
        .into_iter()
        .find(|cand| crate::platform::resolve_llama_binary(cand, "llama-server").is_some())
}
