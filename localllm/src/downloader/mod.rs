//! Model downloader.
//!   * [`hf_api`] — low-level HuggingFace API client (metadata + file fetch).
//!   * [`file_manager`] — top-level pull orchestration (parallel, retry, manifest).

pub mod file_manager;
pub mod hf_api;

pub use file_manager::{DownloadManager, PullProgress};
pub use hf_api::HuggingFaceClient;
