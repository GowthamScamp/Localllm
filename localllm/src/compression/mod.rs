//! Model quantization (safetensors → GGUF via llama.cpp tooling).

pub mod quantize;
pub use quantize::CompressionEngine;
