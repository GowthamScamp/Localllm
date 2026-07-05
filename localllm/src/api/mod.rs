//! HTTP API surface for the daemon.
//!   * [`routes`] — all endpoint handlers and the router builder.
//!   * [`types`] — request/response wire types (OpenAI + Ollama + native).
//!   * [`middleware`] — request-ID injection and tracing span setup.

pub mod middleware;
pub mod routes;
pub mod types;
