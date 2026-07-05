//! Domain error type for localllm. Most internal code uses `anyhow::Result`
//! for convenience, but this `LocalLlmError` enum exists for places that want
//! to discriminate error kinds (e.g. mapping to HTTP status codes, or matching
//! on a specific category like `Download` vs `Gpu`).

use thiserror::Error;

/// Tagged union of failure categories the daemon can produce.
///
/// Variants ending in `(String)` are catch-all wrappers — they carry a human
/// message but no structured detail. The `#[from]` variants are auto-converted
/// from their wrapped error type, so I/O / HTTP / JSON failures don't need
/// boilerplate `.map_err(...)` at every call site.
#[derive(Error, Debug)]
pub enum LocalLlmError {
    #[error("Download error: {0}")]
    Download(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Registry error: {0}")]
    Registry(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
