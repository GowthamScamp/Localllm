//! Model registry.
//!   * [`manifest`] — `ModelManifest` and disk-backed `ManifestStore` cache.
//!   * [`modelfile`] — Ollama-style Modelfile parser (SYSTEM / PARAMETER / etc).

pub mod manifest;
pub mod modelfile;

pub use manifest::{ManifestStore, ModelManifest, QuantizationLevel};
pub use modelfile::Modelfile;
