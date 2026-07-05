//! # HTTP route handlers
//!
//! Single Axum `Router` that exposes three API surfaces simultaneously:
//!   * **OpenAI** `/v1/*` — chat completions, completions, embeddings, models.
//!   * **Ollama** `/api/{tags,show,generate,chat,...}` — Ollama wire format.
//!   * **Native** `/api/{pull,quantize,load,disk-usage,gc,ps,...}` — used by
//!     the localllm CLI.
//!
//! Most chat/completion handlers don't deserialize the JSON body — they accept
//! raw `Bytes`, parse only the `model` and `stream` fields, then proxy the rest
//! verbatim to the upstream inference backend. This avoids a roundtrip
//! deserialize→reserialize on the hot path and means new OpenAI params work
//! transparently without code changes here.

use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use bytes::Bytes;
use chrono::TimeZone;
use futures::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::middleware::request_id_middleware;
use crate::api::types::*;
use crate::compression::CompressionEngine;
use crate::config::Settings;
use crate::downloader::DownloadManager;
use crate::inference::{InferenceRouter, LlamaCppManager, SglangManager};
use crate::registry::{ManifestStore, ModelManifest, Modelfile, QuantizationLevel};

/// Process-wide counters for /metrics. AtomicU64 → lock-free updates.
#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub chat_requests_total: AtomicU64,
    pub completion_requests_total: AtomicU64,
    pub embeddings_requests_total: AtomicU64,
    pub model_loads_total: AtomicU64,
    pub model_pulls_total: AtomicU64,
    pub bytes_proxied_total: AtomicU64,
    pub errors_total: AtomicU64,
    /// Total tokens generated across all chat/completion streams. Counted by
    /// scanning each SSE `delta.content` chunk and approximating one token per
    /// whitespace-separated word — adequate for throughput trending.
    pub tokens_generated_total: AtomicU64,
    /// Total prompt tokens reported by upstream in `usage.prompt_tokens` (final
    /// SSE event or non-streaming response body).
    pub prompt_tokens_total: AtomicU64,
}

/// Shared state injected into every handler via `axum::extract::State`.
///
/// Every field is cheap to clone (`Arc<T>` or `reqwest::Client` which clones
/// its inner Arc). `compression_engine` is `Option` because llama.cpp is an
/// optional dependency — without it, the quantize endpoint returns a clean
/// error instead of crashing.
pub struct AppState {
    pub settings: Arc<Settings>,
    pub registry: Arc<ManifestStore>,
    pub download_manager: Arc<DownloadManager>,
    pub compression_engine: Option<Arc<CompressionEngine>>,
    pub inference_router: Arc<InferenceRouter>,
    pub sglang_manager: Arc<SglangManager>,
    pub llamacpp_manager: Arc<LlamaCppManager>,
    pub http_client: reqwest::Client,
    pub metrics: Arc<Metrics>,
    /// Periodic-health-check results, keyed by alias. Updated by a background
    /// task that pings each running backend's `/health` every 30s; read by
    /// `/health` to expose an `unhealthy` list of currently-failing backends.
    pub backend_health: Arc<dashmap::DashMap<String, BackendHealth>>,
}

#[derive(Debug, Clone)]
pub struct BackendHealth {
    pub healthy: bool,
    pub last_checked: i64,
    pub last_error: Option<String>,
}

/// Build the full Axum router with every endpoint mounted and middleware layered.
///
/// Middleware order (outermost to innermost):
///   1. `TraceLayer` — tracing-span-per-request.
///   2. `CorsLayer::permissive` — allow any origin (fine for localhost).
///   3. `request_id_middleware` — injects/echoes `X-Request-ID`.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // OpenAI-compatible
        .route("/v1/models", get(list_models_v1))
        .route("/v1/models/:id", get(retrieve_model_v1))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings_openai))
        // Ollama-compatible
        .route("/api/version", get(ollama_version))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/embeddings", post(ollama_embeddings))
        .route("/api/copy", post(ollama_copy))
        .route("/api/create", post(ollama_create))
        .route("/api/delete", delete(ollama_delete))
        // Native localllm management
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/api/pull", post(pull_model))
        .route("/api/quantize", post(quantize_model))
        .route("/api/load", post(load_model))
        .route("/api/models", get(list_models_api))
        .route("/api/models/:alias", delete(delete_model))
        .route("/api/ps", get(ps_handler))
        .route("/api/logs/:alias", get(logs_handler))
        .route("/api/disk-usage", get(disk_usage_handler))
        .route("/api/gc", post(gc_handler))
        .with_state(state)
        .layer(middleware::from_fn(request_id_middleware))
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

// =============================================================================
// Trivial handlers
// =============================================================================

/// `GET /health` — liveness check. Returns JSON with version + counts.
/// Used by the auto-spawn polling loop in main.rs and by external monitors.
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let sglang_count = state.sglang_manager.processes.len();
    let llamacpp_count = state.llamacpp_manager.processes.len();
    let model_count = state.registry.list().len();

    // Build the unhealthy list from cached backend_health probes.
    let unhealthy: Vec<serde_json::Value> = state
        .backend_health
        .iter()
        .filter(|e| !e.value().healthy)
        .map(|e| {
            serde_json::json!({
                "alias": e.key().clone(),
                "last_checked": e.value().last_checked,
                "error": e.value().last_error.clone(),
            })
        })
        .collect();

    let overall_status = if unhealthy.is_empty() { "ok" } else { "degraded" };

    Json(serde_json::json!({
        "status": overall_status,
        "version": "0.1.0",
        "models_total": model_count,
        "models_loaded": sglang_count + llamacpp_count,
        "sglang_processes": sglang_count,
        "llamacpp_processes": llamacpp_count,
        "unhealthy": unhealthy,
    }))
}

/// `GET /metrics` — Prometheus exposition format. Plain text, no auth.
///
/// Each counter is hand-formatted with `# HELP` and `# TYPE` lines per
/// Prometheus convention. We expose request counts per endpoint family,
/// model lifecycle events, bytes proxied, and the current live process gauge.
///
/// Scrape with `prometheus.yml`: `targets: ['127.0.0.1:11435']`.
async fn metrics_handler(State(state): State<Arc<AppState>>) -> Response {
    let m = &state.metrics;
    let body = format!(
        concat!(
            "# HELP localllm_requests_total Total HTTP requests received\n",
            "# TYPE localllm_requests_total counter\n",
            "localllm_requests_total {}\n",
            "# HELP localllm_chat_requests_total Total chat completion requests\n",
            "# TYPE localllm_chat_requests_total counter\n",
            "localllm_chat_requests_total {}\n",
            "# HELP localllm_completion_requests_total Total text completion requests\n",
            "# TYPE localllm_completion_requests_total counter\n",
            "localllm_completion_requests_total {}\n",
            "# HELP localllm_embeddings_requests_total Total embeddings requests\n",
            "# TYPE localllm_embeddings_requests_total counter\n",
            "localllm_embeddings_requests_total {}\n",
            "# HELP localllm_model_loads_total Total model load events\n",
            "# TYPE localllm_model_loads_total counter\n",
            "localllm_model_loads_total {}\n",
            "# HELP localllm_model_pulls_total Total model pull events\n",
            "# TYPE localllm_model_pulls_total counter\n",
            "localllm_model_pulls_total {}\n",
            "# HELP localllm_bytes_proxied_total Bytes proxied through inference endpoints\n",
            "# TYPE localllm_bytes_proxied_total counter\n",
            "localllm_bytes_proxied_total {}\n",
            "# HELP localllm_errors_total Errors returned to clients\n",
            "# TYPE localllm_errors_total counter\n",
            "localllm_errors_total {}\n",
            "# HELP localllm_tokens_generated_total Approx generation tokens streamed back to clients\n",
            "# TYPE localllm_tokens_generated_total counter\n",
            "localllm_tokens_generated_total {}\n",
            "# HELP localllm_prompt_tokens_total Prompt tokens reported by upstream usage field\n",
            "# TYPE localllm_prompt_tokens_total counter\n",
            "localllm_prompt_tokens_total {}\n",
            "# HELP localllm_models_loaded Currently loaded inference processes\n",
            "# TYPE localllm_models_loaded gauge\n",
            "localllm_models_loaded {}\n",
        ),
        m.requests_total.load(Ordering::Relaxed),
        m.chat_requests_total.load(Ordering::Relaxed),
        m.completion_requests_total.load(Ordering::Relaxed),
        m.embeddings_requests_total.load(Ordering::Relaxed),
        m.model_loads_total.load(Ordering::Relaxed),
        m.model_pulls_total.load(Ordering::Relaxed),
        m.bytes_proxied_total.load(Ordering::Relaxed),
        m.errors_total.load(Ordering::Relaxed),
        m.tokens_generated_total.load(Ordering::Relaxed),
        m.prompt_tokens_total.load(Ordering::Relaxed),
        state.sglang_manager.processes.len() + state.llamacpp_manager.processes.len(),
    );
    Response::builder()
        .status(200)
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

async fn list_models_v1(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    let manifests = state.registry.list();
    let data: Vec<ModelObject> = manifests
        .iter()
        .map(|m| ModelObject::new(m.alias.clone(), m.downloaded_at.timestamp()))
        .collect();
    Json(ModelListResponse::new(data))
}

/// OpenAI `GET /v1/models/:id` — retrieve a single model by id.
/// Some clients (e.g. LangChain, certain SDKs) probe this before chat calls.
async fn retrieve_model_v1(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let m = find_manifest(&state, &id)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", id)))?;
    Ok(Json(ModelObject::new(m.alias.clone(), m.downloaded_at.timestamp())))
}

/// Ollama `GET /api/version` — clients probe this for capability detection.
/// We pin a known-compatible Ollama version string so clients don't refuse to talk.
async fn ollama_version() -> impl IntoResponse {
    Json(serde_json::json!({
        "version": "0.5.0"
    }))
}

/// Ollama `DELETE /api/delete` — body is `{"name": "<alias>"}`.
/// Wraps the existing native deletion path so client behavior matches Ollama.
async fn ollama_delete(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ApiError> {
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("Missing 'name' field"))?
        .to_string();
    delete_model_inner(&state, &name).await?;
    Ok(StatusCode::OK)
}

// =============================================================================
// Pull / Quantize / Load / Delete
// =============================================================================

/// `POST /api/pull` — dispatches on body shape:
///   - native localllm CLI sends `{"repo_id": "...", "revision": ..., "quantize": ...}`
///     and gets back the saved manifest as JSON.
///   - Ollama clients send `{"name": "...", "stream": true}` and expect Ollama-style
///     NDJSON progress frames: `{"status": "pulling manifest"}\n{"status": "success"}\n`.
async fn pull_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    state.metrics.model_pulls_total.fetch_add(1, Ordering::Relaxed);

    // Ollama shape: has "name", lacks "repo_id"
    let is_ollama_shape = body.get("name").is_some() && body.get("repo_id").is_none();

    if is_ollama_shape {
        let name = body
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ApiError::bad_request("Missing 'name' field"))?
            .to_string();
        let stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        return ollama_pull_stream(state, name, stream).await;
    }

    let req: PullRequest = serde_json::from_value(body)
        .map_err(|e| ApiError::bad_request(format!("Invalid pull request: {}", e)))?;
    tracing::info!("Pull request: {}", req.repo_id);

    if req.stream == Some(true) {
        return native_pull_stream(state, req).await;
    }

    let dl = download_manager_for(&state, req.hf_token.as_deref())?;
    let manifest = pull_and_optionally_quantize(&state, &dl, &req.repo_id, req.revision.as_deref(), req.quantize.as_deref()).await?;
    Ok(Json(manifest).into_response())
}

/// Resolve the `DownloadManager` to use for a pull. When the request carries an
/// `hf_token`, build a one-off manager with an authenticated HuggingFace client
/// (gated models work, and authenticated requests get higher rate limits →
/// faster downloads). Otherwise reuse the daemon's shared manager, which was
/// built from the daemon's own `HF_TOKEN` env at startup.
fn download_manager_for(
    state: &AppState,
    hf_token: Option<&str>,
) -> Result<Arc<DownloadManager>, ApiError> {
    match hf_token {
        Some(tok) if !tok.is_empty() => {
            let client = crate::downloader::HuggingFaceClient::new(Some(tok.to_string()))
                .map_err(|e| ApiError::internal(format!("Failed to build HF client: {}", e)))?;
            Ok(Arc::new(DownloadManager::new(
                Arc::new(client),
                state.settings.clone(),
            )))
        }
        _ => Ok(state.download_manager.clone()),
    }
}

/// Whether this machine has at least one usable NVIDIA GPU. Non-fatal: any probe
/// failure (no driver, no `nvidia-smi`) is treated as "no GPU".
async fn has_gpu() -> bool {
    crate::gpu::VramManager::query_gpus_async()
        .await
        .map(|g| !g.is_empty())
        .unwrap_or(false)
}

/// Decide what a pull should actually fetch.
///
/// * An explicit `--quantize` always wins → fetch/convert that GGUF level.
/// * No quant + **no GPU** → default to a runnable `Q4_K_M` GGUF. Raw safetensors
///   can't run on CPU (llama.cpp needs GGUF; converting needs Python+torch), so a
///   plain `pull` on a CPU box must yield something that actually runs.
/// * No quant + **GPU present** → keep `None` (pull full-precision safetensors for
///   the sglang/GPU path).
///
/// Returns the quant level string to use, or `None` to pull safetensors.
async fn effective_pull_quant(requested: Option<&str>) -> Option<String> {
    if let Some(q) = requested {
        return Some(q.to_string());
    }
    if has_gpu().await {
        None
    } else {
        Some("Q4_K_M".to_string())
    }
}

/// Shared core: download (+ optionally quantize) + save manifest. Used by the
/// non-streaming native and Ollama-shaped `/api/pull` paths.
///
/// Strategy (mirrors the streaming path so behavior is consistent):
///   * Compute the effective quant — explicit `--quantize` wins; otherwise default
///     to a runnable `Q4_K_M` GGUF on a CPU-only box, or safetensors when a GPU is
///     present (sglang path).
///   * When a quant is wanted, prefer a **prebuilt GGUF** (torch-free); fall back to
///     safetensors + local convert only if no prebuilt GGUF exists AND a compression
///     engine (llama_cpp_dir + Python/torch) is configured.
///   * When no quant (GPU present), pull full-precision safetensors.
async fn pull_and_optionally_quantize(
    state: &AppState,
    download_manager: &DownloadManager,
    repo_id: &str,
    revision: Option<&str>,
    quantize: Option<&str>,
) -> Result<ModelManifest, ApiError> {
    let effective = effective_pull_quant(quantize).await;

    if let Some(quant_str) = effective {
        let level = QuantizationLevel::from_str(&quant_str).ok_or_else(|| {
            ApiError::bad_request(format!("Unknown quantization level: {}", quant_str))
        })?;

        // 1) Prefer a prebuilt GGUF — no torch, exactly how Ollama works.
        match download_manager
            .pull_gguf(repo_id, &quant_str, |_p| {})
            .await
        {
            Ok(manifest) => {
                state
                    .registry
                    .save(&manifest)
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                return Ok(manifest);
            }
            Err(gguf_err) => {
                // 2) Fall back to safetensors + local convert, only if we can.
                let engine = state.compression_engine.as_ref().ok_or_else(|| {
                    ApiError::internal(format!(
                        "No prebuilt GGUF found for '{}' and local conversion is unavailable: {}",
                        repo_id, gguf_err
                    ))
                    .with_hint(
                        "Set llama_cpp_dir + install Python/torch to convert, or pick a model with a GGUF release",
                    )
                })?;
                let mut manifest = download_manager
                    .pull(repo_id, revision)
                    .await
                    .map_err(|e| {
                        ApiError::internal(e.to_string())
                            .with_hint("Verify HF_TOKEN if pulling a private model")
                    })?;
                let gguf_path = engine
                    .quantize(&manifest, level.clone())
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                manifest.gguf_path = Some(gguf_path);
                manifest.quantization = Some(level);
                state
                    .registry
                    .save(&manifest)
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                return Ok(manifest);
            }
        }
    }

    // No quant (GPU present): pull full-precision safetensors for sglang.
    let manifest = download_manager
        .pull(repo_id, revision)
        .await
        .map_err(|e| {
            ApiError::internal(e.to_string())
                .with_hint("Verify HF_TOKEN if pulling a private model")
        })?;
    state
        .registry
        .save(&manifest)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(manifest)
}

/// Ollama-shaped streaming pull. Emits one NDJSON frame per phase. The actual
/// download runs as one atomic operation under the hood, so we synthesize phase
/// frames — clients only need them to display status; per-byte progress would
/// require deeper plumbing into the downloader.
async fn ollama_pull_stream(
    state: Arc<AppState>,
    name: String,
    stream: bool,
) -> Result<Response, ApiError> {
    if !stream {
        let manifest =
            pull_and_optionally_quantize(&state, &state.download_manager, &name, None, None).await?;
        let total: u64 = manifest.files.iter().map(|f| f.size_bytes).sum();
        return Ok(Json(serde_json::json!({
            "status": "success",
            "digest": manifest.revision,
            "total": total,
        }))
        .into_response());
    }

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);

    // Phase 1: send "pulling manifest" immediately so the client sees activity.
    let _ = tx
        .send(Ok(ndjson_frame(&serde_json::json!({"status": "pulling manifest"}))))
        .await;

    let state_clone = state.clone();
    let name_clone = name.clone();
    tokio::spawn(async move {
        // Phase 2: do the actual pull. Long-running.
        let dm = state_clone.download_manager.clone();
        let pull_result =
            pull_and_optionally_quantize(&state_clone, &dm, &name_clone, None, None).await;

        match pull_result {
            Ok(manifest) => {
                let total: u64 = manifest.files.iter().map(|f| f.size_bytes).sum();
                // Synthesize a "pulling <digest>" + completion frame per file so
                // clients that count files see the right number.
                for f in &manifest.files {
                    let frame = serde_json::json!({
                        "status": format!("pulling {}", f.name),
                        "digest": f.sha256,
                        "total": f.size_bytes,
                        "completed": f.size_bytes,
                    });
                    if tx.send(Ok(ndjson_frame(&frame))).await.is_err() {
                        return;
                    }
                }
                let _ = tx
                    .send(Ok(ndjson_frame(&serde_json::json!({
                        "status": "verifying sha256 digest"
                    }))))
                    .await;
                let _ = tx
                    .send(Ok(ndjson_frame(&serde_json::json!({
                        "status": "writing manifest"
                    }))))
                    .await;
                let _ = tx
                    .send(Ok(ndjson_frame(&serde_json::json!({
                        "status": "success",
                        "digest": manifest.revision,
                        "total": total,
                    }))))
                    .await;
            }
            Err(e) => {
                let _ = tx
                    .send(Ok(ndjson_frame(&serde_json::json!({
                        "status": "error",
                        "error": e.error.message,
                    }))))
                    .await;
            }
        }
    });

    Response::builder()
        .status(200)
        .header("Content-Type", "application/x-ndjson")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .map_err(|e| ApiError::internal(e.to_string()))
}

/// Native streaming pull. Sends NDJSON frames so the CLI can render a live
/// progress bar instead of silently waiting. Strategy:
///
///   * If a quant level is requested (the common CPU case), try to download a
///     **prebuilt GGUF** directly — no torch, no conversion. Emits real
///     byte-level progress frames as the file streams in.
///   * If no GGUF exists anywhere, fall back to the safetensors pull +
///     `convert_hf_to_gguf.py` path (requires torch + llama_cpp_dir).
///   * If no quant requested, do a plain safetensors pull (for GPU/sglang).
///
/// Frame shapes:
///   {"status": "<phase>"}
///   {"status": "download", "downloaded": N, "total": M}
///   {"status": "success", "alias": "...", ...}
///   {"status": "error", "error": "..."}
async fn native_pull_stream(
    state: Arc<AppState>,
    req: PullRequest,
) -> Result<Response, ApiError> {
    use crate::downloader::PullProgress;

    // Resolve the download manager up-front so a malformed token fails fast with
    // a clean HTTP error instead of mid-stream. Carries the request's hf_token
    // for authenticated (faster, gated-capable) downloads.
    let download_manager = download_manager_for(&state, req.hf_token.as_deref())?;

    // Decide what to actually fetch: an explicit quant wins; otherwise default to
    // a runnable Q4_K_M GGUF on a CPU-only box (raw safetensors can't run there),
    // or safetensors when a GPU is present (sglang path).
    let mut req = req;
    req.quantize = effective_pull_quant(req.quantize.as_deref()).await;

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let _ = tx
        .send(Ok(ndjson_frame(&serde_json::json!({"status": "pulling manifest"}))))
        .await;

    let state_clone = state.clone();
    tokio::spawn(async move {
        let send_err = |tx: &mpsc::Sender<Result<Bytes, std::io::Error>>, msg: String| {
            let frame = ndjson_frame(&serde_json::json!({"status": "error", "error": msg}));
            let tx = tx.clone();
            async move {
                let _ = tx.send(Ok(frame)).await;
            }
        };

        // ---- GGUF auto-download path (preferred when a quant is requested) ----
        if let Some(quant) = req.quantize.clone() {
            // Throttle byte-progress frames so we don't flood the channel — at
            // most one frame per ~250ms or per 1% of total, whichever first.
            let tx_p = tx.clone();
            let last_emit = std::sync::Mutex::new(std::time::Instant::now());
            let on_progress = move |p: PullProgress| {
                let frame = match p {
                    PullProgress::Status(s) => {
                        Some(ndjson_frame(&serde_json::json!({"status": s})))
                    }
                    PullProgress::Download { downloaded, total } => {
                        let mut guard = last_emit.lock().unwrap();
                        let due = guard.elapsed() >= std::time::Duration::from_millis(250)
                            || (total > 0 && downloaded >= total);
                        if due {
                            *guard = std::time::Instant::now();
                            Some(ndjson_frame(&serde_json::json!({
                                "status": "download",
                                "downloaded": downloaded,
                                "total": total,
                            })))
                        } else {
                            None
                        }
                    }
                };
                if let Some(f) = frame {
                    // Best-effort: blocking_send would panic in async; use try_send.
                    let _ = tx_p.try_send(Ok(f));
                }
            };

            let result = download_manager
                .pull_gguf(&req.repo_id, &quant, on_progress)
                .await;

            match result {
                Ok(manifest) => {
                    if let Err(e) = state_clone.registry.save(&manifest) {
                        send_err(&tx, e.to_string()).await;
                        return;
                    }
                    let _ = tx
                        .send(Ok(ndjson_frame(&serde_json::json!({
                            "status": "success",
                            "alias": manifest.alias,
                            "architecture": manifest.architecture,
                            "parameters_billion": manifest.parameters_billion,
                            "context_length": manifest.context_length,
                            "quantization": manifest.quantization.as_ref().map(|q| q.to_llama_str()),
                        }))))
                        .await;
                    return;
                }
                Err(gguf_err) => {
                    // No prebuilt GGUF. Fall back to safetensors + convert only if
                    // a compression engine (llama_cpp_dir) is configured.
                    tracing::info!("GGUF auto-download failed ({}); trying convert path", gguf_err);
                    let _ = tx
                        .send(Ok(ndjson_frame(&serde_json::json!({
                            "status": "no prebuilt GGUF — converting from safetensors"
                        }))))
                        .await;

                    match pull_and_convert(&state_clone, &download_manager, &req, &quant, &tx).await {
                        Ok(manifest) => {
                            let _ = tx
                                .send(Ok(ndjson_frame(&serde_json::json!({
                                    "status": "success",
                                    "alias": manifest.alias,
                                    "architecture": manifest.architecture,
                                    "parameters_billion": manifest.parameters_billion,
                                    "context_length": manifest.context_length,
                                    "quantization": manifest.quantization.as_ref().map(|q| q.to_llama_str()),
                                }))))
                                .await;
                        }
                        Err(e) => {
                            send_err(
                                &tx,
                                format!(
                                    "{}. Also failed safetensors conversion: {}",
                                    gguf_err, e
                                ),
                            )
                            .await;
                        }
                    }
                    return;
                }
            }
        }

        // ---- No quant requested: plain safetensors pull (GPU path) ----
        let _ = tx
            .send(Ok(ndjson_frame(&serde_json::json!({"status": "downloading safetensors"}))))
            .await;
        match download_manager
            .pull(&req.repo_id, req.revision.as_deref())
            .await
        {
            Ok(manifest) => {
                if let Err(e) = state_clone.registry.save(&manifest) {
                    send_err(&tx, e.to_string()).await;
                    return;
                }
                let _ = tx
                    .send(Ok(ndjson_frame(&serde_json::json!({
                        "status": "success",
                        "alias": manifest.alias,
                        "architecture": manifest.architecture,
                        "parameters_billion": manifest.parameters_billion,
                        "context_length": manifest.context_length,
                        "quantization": serde_json::Value::Null,
                    }))))
                    .await;
            }
            Err(e) => send_err(&tx, e.to_string()).await,
        }
    });

    Response::builder()
        .status(200)
        .header("Content-Type", "application/x-ndjson")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .map_err(|e| ApiError::internal(e.to_string()))
}

/// Fallback: pull safetensors, then run the convert+quantize pipeline. Returns
/// the saved manifest. Requires `compression_engine` (llama_cpp_dir) + torch.
async fn pull_and_convert(
    state: &AppState,
    download_manager: &DownloadManager,
    req: &PullRequest,
    quant: &str,
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<ModelManifest, anyhow::Error> {
    let mut manifest = download_manager
        .pull(&req.repo_id, req.revision.as_deref())
        .await?;

    let level = QuantizationLevel::from_str(quant)
        .ok_or_else(|| anyhow::anyhow!("Unknown quantization level: {}", quant))?;
    let engine = state
        .compression_engine
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("llama_cpp_dir not configured for conversion"))?;

    let _ = tx
        .send(Ok(ndjson_frame(&serde_json::json!({
            "status": format!("quantizing to {}", quant)
        }))))
        .await;

    let gguf_path = engine.quantize(&manifest, level.clone()).await?;
    manifest.gguf_path = Some(gguf_path);
    manifest.quantization = Some(level);
    state.registry.save(&manifest)?;
    Ok(manifest)
}

fn ndjson_frame(v: &serde_json::Value) -> Bytes {
    let mut buf = serde_json::to_vec(v).unwrap_or_default();
    buf.push(b'\n');
    Bytes::from(buf)
}

/// `POST /api/quantize` — convert an already-pulled safetensors model to GGUF
/// at the requested precision. Two-stage: convert_hf_to_gguf.py to F16, then
/// llama-quantize to the target level. Updates the manifest in place.
async fn quantize_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QuantizeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let manifest_arc = state.registry.get(&req.alias).ok_or_else(|| {
        ApiError::not_found(format!("Model not found: {}", req.alias))
            .with_hint("Run: localllm pull <repo_id>")
    })?;
    let mut manifest = (*manifest_arc).clone();

    let level = QuantizationLevel::from_str(&req.level).ok_or_else(|| {
        ApiError::bad_request(format!("Unknown quantization level: {}", req.level))
    })?;

    let engine = state
        .compression_engine
        .as_ref()
        .ok_or_else(|| {
            ApiError::internal("Compression engine not available")
                .with_hint("Set llama_cpp_dir in ~/.localllm/config.toml")
        })?;

    let gguf_path = engine
        .quantize(&manifest, level.clone())
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    manifest.gguf_path = Some(gguf_path);
    manifest.quantization = Some(level);

    state
        .registry
        .save(&manifest)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(manifest))
}

/// `POST /api/load` — eagerly start the inference backend for a model so the
/// first user request doesn't pay startup latency. Returns the endpoint URL
/// so clients can probe it directly if they want.
async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let manifest = state
        .registry
        .get(&req.alias)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", req.alias)))?;

    let endpoint = state
        .inference_router
        .get_endpoint(&manifest)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    state.metrics.model_loads_total.fetch_add(1, Ordering::Relaxed);

    Ok(Json(serde_json::json!({
        "status": "loaded",
        "endpoint": endpoint
    })))
}

/// `DELETE /api/models/:alias` — native delete endpoint. Returns 204 No Content
/// on success. The Ollama-style `DELETE /api/delete` delegates to the same
/// implementation via `delete_model_inner` and returns 200 OK instead.
async fn delete_model(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    delete_model_inner(&state, &alias).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Shared deletion logic. Steps:
///   1. Look up manifest (fails with 404 if absent).
///   2. Kill any running sglang/llama-server for this alias (best-effort).
///   3. Recursively remove the safetensors directory.
///   4. Remove the GGUF file if one exists.
///   5. Drop the manifest entry from the cache and disk.
async fn delete_model_inner(state: &AppState, alias: &str) -> Result<(), ApiError> {
    let manifest = state
        .registry
        .get(alias)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", alias)))?;

    let _ = state.sglang_manager.kill(alias).await;
    let _ = state.llamacpp_manager.kill(alias).await;

    if manifest.local_path.exists() {
        tokio::fs::remove_dir_all(&manifest.local_path)
            .await
            .map_err(|e| ApiError::internal(format!("Failed to remove model dir: {}", e)))?;
    }

    if let Some(gguf_path) = &manifest.gguf_path {
        if gguf_path.exists() {
            tokio::fs::remove_file(gguf_path)
                .await
                .map_err(|e| ApiError::internal(format!("Failed to remove GGUF: {}", e)))?;
        }
    }

    state
        .registry
        .delete(alias)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    tracing::info!("Deleted model: {}", alias);
    Ok(())
}

/// `GET /api/models` — native listing with status per model.
/// Status is `"loaded"` if an inference process is alive for this alias,
/// else `"ready"` (manifest exists, model not loaded).
async fn list_models_api(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let manifests = state.registry.list();
    let entries: Vec<ModelStatusEntry> = manifests
        .iter()
        .map(|m| {
            let status = if state.sglang_manager.processes.contains_key(&m.alias)
                || state.llamacpp_manager.processes.contains_key(&m.alias)
            {
                "loaded"
            } else {
                "ready"
            };
            ModelStatusEntry {
                alias: m.alias.clone(),
                repo_id: m.repo_id.clone(),
                architecture: m.architecture.clone(),
                parameters_billion: m.parameters_billion,
                quantization: m.quantization.as_ref().map(|q| q.to_llama_str().to_string()),
                weight_format: format!("{:?}", m.weight_format),
                status: status.to_string(),
                gguf_path: m
                    .gguf_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
            }
        })
        .collect();
    Json(entries)
}

/// `GET /api/ps` — list running inference processes (alias, backend, port, age).
/// Merges sglang and llama-server tables; each backend is reported separately
/// even if both happen to be running for the same alias (rare but possible).
async fn ps_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut entries = Vec::new();
    for entry in state.sglang_manager.processes.iter() {
        let p = entry.value();
        let ts = p.last_used.load(Ordering::Relaxed);
        let last_used = chrono::Utc
            .timestamp_opt(ts, 0)
            .single()
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default();
        entries.push(PsEntry {
            alias: p.model_alias.clone(),
            port: p.port,
            backend: "sglang".to_string(),
            started_at: p.started_at.to_rfc3339(),
            last_used,
        });
    }
    for entry in state.llamacpp_manager.processes.iter() {
        let p = entry.value();
        let ts = p.last_used.load(Ordering::Relaxed);
        let last_used = chrono::Utc
            .timestamp_opt(ts, 0)
            .single()
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default();
        entries.push(PsEntry {
            alias: p.model_alias.clone(),
            port: p.port,
            backend: "llamacpp".to_string(),
            started_at: p.started_at.to_rfc3339(),
            last_used,
        });
    }
    Json(entries)
}

/// `GET /api/logs/:alias?lines=N` — last `N` (default 100) stdout/stderr lines
/// captured from the running inference process for `alias`. Returns 404 if no
/// process is currently loaded for that alias.
async fn logs_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
    axum::extract::Query(q): axum::extract::Query<LogsQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let lines_req = q.lines.unwrap_or(100);

    // sglang first, fall back to llamacpp
    if let Some(entry) = state.sglang_manager.processes.get(&alias) {
        let lines = entry.value().logs.snapshot(lines_req);
        return Ok(Json(serde_json::json!({
            "alias": alias,
            "backend": "sglang",
            "lines": lines,
        })));
    }
    if let Some(entry) = state.llamacpp_manager.processes.get(&alias) {
        let lines = entry.value().logs.snapshot(lines_req);
        return Ok(Json(serde_json::json!({
            "alias": alias,
            "backend": "llamacpp",
            "lines": lines,
        })));
    }
    Err(ApiError::not_found(format!(
        "No running inference process for '{}'",
        alias
    ))
    .with_hint("Run: localllm load <alias>"))
}

#[derive(serde::Deserialize)]
struct LogsQuery {
    lines: Option<usize>,
}

// =============================================================================
// /v1/chat/completions, /v1/completions, /v1/embeddings (OpenAI)
// =============================================================================

/// `POST /v1/chat/completions` — proxy to the chat endpoint of the inference
/// backend chosen for `model`. Streams SSE if `stream:true` in the body.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ApiError> {
    state.metrics.chat_requests_total.fetch_add(1, Ordering::Relaxed);
    proxy_openai_chat_or_completion(&state, body, "/v1/chat/completions", true).await
}

/// `POST /v1/completions` — same as `chat_completions` but for text completion
/// (legacy OpenAI prompt-only endpoint). `is_chat=false` skips Modelfile
/// SYSTEM injection since text completions have no message structure.
async fn completions(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ApiError> {
    state.metrics.completion_requests_total.fetch_add(1, Ordering::Relaxed);
    proxy_openai_chat_or_completion(&state, body, "/v1/completions", false).await
}

/// `POST /v1/embeddings` — proxy to the embeddings endpoint. Must ensure the
/// model was started with the `--embeddings` flag (llama.cpp requirement);
/// if not, we set the manifest flag, kick the running process, and let the
/// router restart it with the right flag on the next request.
async fn embeddings_openai(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ApiError> {
    state.metrics.embeddings_requests_total.fetch_add(1, Ordering::Relaxed);

    let routing: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("Invalid JSON: {}", e)))?;
    let model = routing
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("Missing 'model' field"))?
        .to_string();

    let manifest_arc = find_manifest(&state, &model)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", model)))?;

    // Ensure model is loaded with the --embeddings flag (only matters for llama.cpp).
    let manifest_arc = if !manifest_arc.embeddings {
        let mut updated = (*manifest_arc).clone();
        updated.embeddings = true;
        state
            .registry
            .save(&updated)
            .map_err(|e| ApiError::internal(e.to_string()))?;
        // Kick any running process so it restarts with the new flag
        let _ = state.llamacpp_manager.kill(&model).await;
        let _ = state.sglang_manager.kill(&model).await;
        Arc::new(updated)
    } else {
        manifest_arc
    };

    let endpoint = state
        .inference_router
        .get_endpoint(&manifest_arc)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    proxy_bytes(&state, format!("{}/v1/embeddings", endpoint), body, false).await
}

/// Core proxy for both `chat_completions` and `completions`. The body is
/// parsed only enough to extract `model` (for routing) and `stream` (for
/// response framing) — everything else passes through to the upstream verbatim.
///
/// If the model has a Modelfile attached, this function rewrites the body
/// in-place to inject SYSTEM/seed MESSAGEs and PARAMETER defaults. Otherwise
/// it forwards the bytes untouched (zero-copy fast path).
async fn proxy_openai_chat_or_completion(
    state: &AppState,
    body: Bytes,
    upstream_path: &str,
    is_chat: bool,
) -> Result<Response, ApiError> {
    // Parse routing fields only — no full schema validation, no re-serialize.
    let routing: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("Invalid JSON: {}", e)))?;
    let model = routing
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("Missing 'model' field"))?
        .to_string();
    let is_stream = routing
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let manifest = find_manifest(state, &model).ok_or_else(|| {
        ApiError::not_found(format!("Model not found: {}", model))
            .with_hint("Run: localllm pull <huggingface-repo>")
    })?;

    // A5 — touch the LRU timestamp in memory (lock-free for the inference
    // process; the manifest copy gets batched to disk every 30s).
    let _ = state.registry.update_last_used(&manifest.alias);

    // Apply Modelfile (SYSTEM + seed messages, PARAMETER defaults) when one exists.
    // We rewrite the body in-place. Otherwise forward the bytes verbatim,
    // preserving the zero-copy fast path.
    let outbound_body: Bytes = match manifest.modelfile.as_ref() {
        Some(mf) => apply_modelfile_to_body(&body, mf, is_chat).unwrap_or(body),
        None => body,
    };

    let endpoint = state
        .inference_router
        .get_endpoint(&manifest)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let url = format!("{}{}", endpoint, upstream_path);
    proxy_bytes(state, url, outbound_body, is_stream).await
}

/// Parse the user's request body, apply Modelfile SYSTEM + seed MESSAGEs (chat only)
/// and PARAMETER defaults, then re-serialize. Returns `None` if anything goes wrong —
/// caller falls back to forwarding the body unchanged.
fn apply_modelfile_to_body(body: &Bytes, mf: &Modelfile, is_chat: bool) -> Option<Bytes> {
    let mut v: serde_json::Value = serde_json::from_slice(body).ok()?;

    if is_chat {
        if let Some(msgs) = v.get_mut("messages").and_then(|m| m.as_array_mut()) {
            let original: Vec<ChatMessage> =
                serde_json::from_value(serde_json::Value::Array(msgs.clone())).ok()?;
            let merged = mf.apply_to_messages(&original);
            *msgs = serde_json::to_value(&merged).ok()?.as_array().cloned()?;
        }
    }

    // Apply PARAMETER defaults — only when the request hasn't already set them.
    // Ollama's mapping: temperature/top_p/top_k → same; num_predict → max_tokens;
    // stop → stop. Everything else is forwarded as-is in case the backend understands it.
    if let Some(obj) = v.as_object_mut() {
        for (k, raw_v) in &mf.parameters {
            let (key, parsed) = match k.as_str() {
                "num_predict" => ("max_tokens".to_string(), parse_modelfile_value(raw_v)),
                "stop" => {
                    // Accumulate multiple `PARAMETER stop "..."` lines into an array.
                    let entry = obj
                        .entry("stop".to_string())
                        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                    if let Some(arr) = entry.as_array_mut() {
                        arr.push(serde_json::Value::String(raw_v.trim_matches('"').to_string()));
                    }
                    continue;
                }
                other => (other.to_string(), parse_modelfile_value(raw_v)),
            };
            obj.entry(key).or_insert(parsed);
        }
    }

    Some(Bytes::from(serde_json::to_vec(&v).ok()?))
}

/// Best-effort parse of a Modelfile value string: number first, then bool, else string.
fn parse_modelfile_value(s: &str) -> serde_json::Value {
    let s = s.trim();
    if let Ok(n) = s.parse::<i64>() {
        return serde_json::Value::Number(n.into());
    }
    if let Ok(n) = s.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(n) {
            return serde_json::Value::Number(num);
        }
    }
    match s {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        _ => serde_json::Value::String(s.trim_matches('"').to_string()),
    }
}

/// Look up a model manifest by alias first, then fall back to repo_id match.
/// This means clients can pass either `"llama-3.2-1b-instruct"` (alias) or
/// `"meta-llama/Llama-3.2-1B-Instruct"` (HF repo_id) in the `model` field.
fn find_manifest(state: &AppState, model: &str) -> Option<Arc<ModelManifest>> {
    state
        .registry
        .get(model)
        .or_else(|| state.registry.find_by_repo_id(model))
}

// =============================================================================
// Ollama-compatible endpoints (B1, B2, B3, B4, B5, B7, B10)
// =============================================================================

/// `GET /api/tags` — Ollama's "list models" endpoint. Returns the same shape
/// Ollama does so its clients (chat UIs, code completion plugins) work unchanged.
async fn ollama_tags(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let manifests = state.registry.list();
    let models: Vec<OllamaModelInfo> = manifests
        .iter()
        .map(|m| {
            let size = m.files.iter().map(|f| f.size_bytes).sum();
            OllamaModelInfo {
                name: m.alias.clone(),
                modified_at: m.last_used.to_rfc3339(),
                size,
                digest: m.revision.clone(),
                details: OllamaModelDetails {
                    format: format!("{:?}", m.weight_format).to_lowercase(),
                    family: m.architecture.clone(),
                    families: vec![m.architecture.clone()],
                    parameter_size: format!("{:.1}B", m.parameters_billion),
                    quantization_level: m
                        .quantization
                        .as_ref()
                        .map(|q| q.to_llama_str().to_string())
                        .unwrap_or_else(|| "F16".to_string()),
                },
            }
        })
        .collect();
    Json(OllamaTagsResponse { models })
}

/// `POST /api/show` — Ollama's "describe one model" endpoint. Returns
/// modelfile source, template, parameters, family/format/quant level, and
/// auxiliary model info (context length, file count, revision).
async fn ollama_show(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaShowRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let m = state
        .registry
        .get(&req.name)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", req.name)))?;

    let modelfile_text = m
        .modelfile
        .as_ref()
        .map(|mf| mf.source.clone())
        .unwrap_or_else(|| format!("FROM {}\n", m.repo_id));
    let template_text = m
        .modelfile
        .as_ref()
        .and_then(|mf| mf.template.clone())
        .unwrap_or_default();
    let params_text = m
        .modelfile
        .as_ref()
        .map(|mf| {
            mf.parameters
                .iter()
                .map(|(k, v)| format!("{} {}", k, v))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    Ok(Json(OllamaShowResponse {
        modelfile: modelfile_text,
        parameters: params_text,
        template: template_text,
        details: OllamaModelDetails {
            format: format!("{:?}", m.weight_format).to_lowercase(),
            family: m.architecture.clone(),
            families: vec![m.architecture.clone()],
            parameter_size: format!("{:.1}B", m.parameters_billion),
            quantization_level: m
                .quantization
                .as_ref()
                .map(|q| q.to_llama_str().to_string())
                .unwrap_or_else(|| "F16".to_string()),
        },
        model_info: serde_json::json!({
            "context_length": m.context_length,
            "parameters_billion": m.parameters_billion,
            "files": m.files.len(),
            "revision": m.revision,
            "downloaded_at": m.downloaded_at.to_rfc3339(),
        }),
    }))
}

/// `POST /api/generate` — Ollama's text-generation endpoint. Translates the
/// Ollama-shaped request into an OpenAI completion request, proxies to the
/// inference backend, then translates the SSE response back to Ollama's
/// NDJSON streaming format.
async fn ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, ApiError> {
    state.metrics.completion_requests_total.fetch_add(1, Ordering::Relaxed);

    let manifest = find_manifest(&state, &req.model)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", req.model)))?;

    let endpoint = state
        .inference_router
        .get_endpoint(&manifest)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let upstream_url = format!("{}/v1/completions", endpoint);

    // If a Modelfile is attached with a TEMPLATE, render the prompt through it
    // (substituting `{{ .System }}` and `{{ .Prompt }}`). Falls back to the raw
    // prompt when no template is defined.
    let rendered_prompt = match manifest.modelfile.as_ref() {
        Some(mf) => mf.render_template(&req.prompt),
        None => req.prompt.clone(),
    };

    // Build OpenAI-style completion request from the Ollama-style one
    let mut upstream_body = serde_json::json!({
        "model": req.model,
        "prompt": rendered_prompt,
        "stream": req.stream,
    });
    if let Some(mf) = manifest.modelfile.as_ref() {
        apply_modelfile_parameters(&mut upstream_body, mf);
    }
    apply_options(&mut upstream_body, &req.options);
    let upstream_bytes = Bytes::from(
        serde_json::to_vec(&upstream_body)
            .map_err(|e| ApiError::internal(e.to_string()))?,
    );

    if req.stream {
        proxy_to_ollama_ndjson(state.clone(), upstream_url, upstream_bytes, req.model, false).await
    } else {
        // Non-stream: collect upstream response, translate to a single NDJSON frame.
        proxy_to_ollama_single(state.clone(), upstream_url, upstream_bytes, req.model, false).await
    }
}

/// `POST /api/chat` — Ollama's chat endpoint. Same shape as `/api/generate`
/// but with a `messages[]` array instead of a flat prompt. Applies Modelfile
/// SYSTEM/seed messages before forwarding upstream.
async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> Result<Response, ApiError> {
    state.metrics.chat_requests_total.fetch_add(1, Ordering::Relaxed);

    let manifest = find_manifest(&state, &req.model)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", req.model)))?;

    // Apply Modelfile if present
    let messages = if let Some(mf) = manifest.modelfile.as_ref() {
        mf.apply_to_messages(&req.messages)
    } else {
        req.messages.clone()
    };

    let endpoint = state
        .inference_router
        .get_endpoint(&manifest)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let upstream_url = format!("{}/v1/chat/completions", endpoint);

    let mut upstream_body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
    });
    if let Some(mf) = manifest.modelfile.as_ref() {
        apply_modelfile_parameters(&mut upstream_body, mf);
    }
    apply_options(&mut upstream_body, &req.options);
    let upstream_bytes = Bytes::from(
        serde_json::to_vec(&upstream_body)
            .map_err(|e| ApiError::internal(e.to_string()))?,
    );

    if req.stream {
        proxy_to_ollama_ndjson(state.clone(), upstream_url, upstream_bytes, req.model, true).await
    } else {
        proxy_to_ollama_single(state.clone(), upstream_url, upstream_bytes, req.model, true).await
    }
}

/// `POST /api/embeddings` — Ollama's embeddings endpoint. Like the OpenAI
/// variant, ensures the model is loaded with `--embeddings` first. Translates
/// the single-vector response back to Ollama's `{"embedding": [...]}` shape.
async fn ollama_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaEmbeddingsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    state.metrics.embeddings_requests_total.fetch_add(1, Ordering::Relaxed);

    let manifest = find_manifest(&state, &req.model)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {}", req.model)))?;

    // Ensure embeddings flag is set on the manifest (llama.cpp needs --embeddings).
    // save() inserts into the cache synchronously so we can reuse `updated_arc`
    // directly — no second registry lookup, no possible race.
    let manifest = if !manifest.embeddings {
        let mut updated = (*manifest).clone();
        updated.embeddings = true;
        state
            .registry
            .save(&updated)
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let _ = state.llamacpp_manager.kill(&req.model).await;
        let _ = state.sglang_manager.kill(&req.model).await;
        Arc::new(updated)
    } else {
        manifest
    };

    let endpoint = state
        .inference_router
        .get_endpoint(&manifest)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let upstream_body = serde_json::json!({
        "model": req.model,
        "input": req.prompt,
    });
    let resp = state
        .http_client
        .post(format!("{}/v1/embeddings", endpoint))
        .json(&upstream_body)
        .send()
        .await
        .map_err(|e| ApiError::internal(format!("Upstream embeddings failed: {}", e)))?;

    let openai: EmbeddingsResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::internal(format!("Failed to parse upstream embeddings: {}", e)))?;

    let embedding = openai
        .data
        .into_iter()
        .next()
        .map(|e| e.embedding)
        .unwrap_or_default();

    Ok(Json(OllamaEmbeddingsResponse { embedding }))
}

/// `POST /api/copy` — alias an existing model under a new name. No file copy
/// happens; both aliases share the same local_path and gguf_path.
async fn ollama_copy(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaCopyRequest>,
) -> Result<impl IntoResponse, ApiError> {
    state
        .registry
        .clone_as(&req.source, &req.destination)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    Ok(StatusCode::OK)
}

/// `POST /api/create` — parse a Modelfile, find its `FROM` base, derive a new
/// manifest that shares the base's weights but carries the SYSTEM/TEMPLATE/
/// PARAMETER customizations. The derived model gets its own alias.
async fn ollama_create(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaCreateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let mf = Modelfile::parse(&req.modelfile)
        .map_err(|e| ApiError::bad_request(format!("Modelfile parse error: {}", e)))?;

    let base_alias = mf
        .from
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("Modelfile missing FROM directive"))?;

    let base = state.registry.get(base_alias).ok_or_else(|| {
        ApiError::not_found(format!("Base model not found: {}", base_alias))
            .with_hint("Run: localllm pull <repo>")
    })?;

    let mut new_manifest = (*base).clone();
    new_manifest.alias = req.name.clone();
    new_manifest.modelfile = Some(mf);
    new_manifest.downloaded_at = chrono::Utc::now();
    new_manifest.last_used = chrono::Utc::now();

    state
        .registry
        .save(&new_manifest)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(serde_json::json!({"status": "success", "name": req.name})))
}

/// Fold Modelfile PARAMETER defaults into the upstream body. The user's
/// `options` (applied after via `apply_options`) still wins — these are defaults
/// only set when the key is missing.
fn apply_modelfile_parameters(upstream: &mut serde_json::Value, mf: &Modelfile) {
    let Some(obj) = upstream.as_object_mut() else { return };
    for (k, raw_v) in &mf.parameters {
        let (key, parsed) = match k.as_str() {
            "num_predict" => ("max_tokens".to_string(), parse_modelfile_value(raw_v)),
            "stop" => {
                let entry = obj
                    .entry("stop".to_string())
                    .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                if let Some(arr) = entry.as_array_mut() {
                    arr.push(serde_json::Value::String(raw_v.trim_matches('"').to_string()));
                }
                continue;
            }
            other => (other.to_string(), parse_modelfile_value(raw_v)),
        };
        obj.entry(key).or_insert(parsed);
    }
}

/// Translate options (Ollama: temperature, top_p, num_predict, num_ctx, stop, ...)
/// into OpenAI-style fields on `upstream_body`. Unknown options are ignored.
fn apply_options(upstream: &mut serde_json::Value, options: &serde_json::Value) {
    if !options.is_object() {
        return;
    }
    let obj = upstream.as_object_mut().expect("upstream is object");
    if let Some(t) = options.get("temperature") {
        obj.insert("temperature".to_string(), t.clone());
    }
    if let Some(p) = options.get("top_p") {
        obj.insert("top_p".to_string(), p.clone());
    }
    if let Some(n) = options.get("num_predict") {
        obj.insert("max_tokens".to_string(), n.clone());
    }
    if let Some(s) = options.get("stop") {
        obj.insert("stop".to_string(), s.clone());
    }
}

// =============================================================================
// Streaming proxy (always-streaming, both branches)
// =============================================================================

/// Generic streaming proxy: forward `body` to `upstream_url`, then stream the
/// response body straight back to the client without buffering.
///
/// Architecture: spawn a background task that reads upstream chunks and
/// forwards them through an `mpsc` channel. The HTTP response body is built
/// from a `ReceiverStream`, so backpressure works naturally — if the client
/// reads slowly, the channel fills and the upstream read pauses.
///
/// Bytes counters (`bytes_proxied_total`) are incremented per chunk so the
/// `/metrics` gauge reflects real throughput, not just request count.
async fn proxy_bytes(
    state: &AppState,
    upstream_url: String,
    body: Bytes,
    is_stream: bool,
) -> Result<Response, ApiError> {
    let body_len = body.len() as u64;
    state
        .metrics
        .bytes_proxied_total
        .fetch_add(body_len, Ordering::Relaxed);

    let mut request = state
        .http_client
        .post(&upstream_url)
        .header("Content-Type", "application/json")
        .body(body);

    if is_stream {
        request = request.header("Accept", "text/event-stream");
    }

    let resp = request.send().await.map_err(|e| {
        state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
        ApiError::internal(format!("Upstream request failed: {}", e))
    })?;

    let upstream_status = resp.status();
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(256);
    let mut byte_stream = resp.bytes_stream();
    let metrics = state.metrics.clone();
    let count_tokens = is_stream;

    tokio::spawn(async move {
        // Cross-chunk buffer for token counting — SSE events can span chunks.
        let mut token_buf: Vec<u8> = Vec::with_capacity(2048);
        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    metrics
                        .bytes_proxied_total
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);

                    // Token-counting on streaming SSE responses. We parse out
                    // each `data: { ... }` event, extract delta.content (or
                    // message.content), approximate one token per whitespace-
                    // delimited word. The estimate matches OpenAI usage to ~10%.
                    if count_tokens {
                        token_buf.extend_from_slice(&bytes);
                        let mut start = 0usize;
                        while let Some(rel) = memchr::memchr(b'\n', &token_buf[start..]) {
                            let end = start + rel;
                            let line = &token_buf[start..end];
                            start = end + 1;
                            count_tokens_in_sse_line(line, &metrics);
                        }
                        // Drain consumed prefix, keep remainder for next chunk.
                        if start > 0 {
                            token_buf.drain(..start);
                        }
                    }

                    if tx.send(Ok(bytes)).await.is_err() {
                        break; // client disconnected
                    }
                }
                Err(e) => {
                    tracing::error!("Upstream stream error: {}", e);
                    let _ = tx
                        .send(Err(std::io::Error::other(
                            e.to_string(),
                        )))
                        .await;
                    break;
                }
            }
        }
    });

    let stream_body = Body::from_stream(ReceiverStream::new(rx));

    let mut builder = Response::builder().status(upstream_status.as_u16());
    if is_stream {
        builder = builder
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("X-Accel-Buffering", "no")
            .header("Connection", "keep-alive");
    } else {
        builder = builder.header("Content-Type", "application/json");
    }

    builder
        .body(stream_body)
        .map_err(|e| ApiError::internal(e.to_string()))
}

/// Proxy an OpenAI SSE response and translate it into Ollama's NDJSON wire
/// format on the fly. Each `data: {...}` event becomes one JSON line; the
/// terminator `data: [DONE]` becomes a final `{"done": true, ...}` line.
///
/// This is the bridge that lets Ollama-style clients talk to OpenAI-style
/// inference backends transparently.
async fn proxy_to_ollama_ndjson(
    state: Arc<AppState>,
    upstream_url: String,
    body: Bytes,
    model: String,
    is_chat: bool,
) -> Result<Response, ApiError> {
    let resp = state
        .http_client
        .post(&upstream_url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .body(body)
        .send()
        .await
        .map_err(|e| ApiError::internal(format!("Upstream failed: {}", e)))?;

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(256);
    let mut byte_stream = resp.bytes_stream();

    tokio::spawn(async move {
        let mut buf = Vec::<u8>::with_capacity(8192);
        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            };
            buf.extend_from_slice(&chunk);
            // Walk SSE lines
            while let Some(pos) = memchr::memchr(b'\n', &buf) {
                let line = buf.drain(..=pos).collect::<Vec<u8>>();
                let line = std::str::from_utf8(&line).unwrap_or("").trim();
                if line.is_empty() {
                    continue;
                }
                if line == "data: [DONE]" {
                    let frame = OllamaStreamFrame {
                        model: model.clone(),
                        created_at: chrono::Utc::now().to_rfc3339(),
                        response: if is_chat { None } else { Some(String::new()) },
                        message: if is_chat {
                            Some(ChatMessage {
                                role: "assistant".into(),
                                content: String::new(),
                            })
                        } else {
                            None
                        },
                        done: true,
                        total_duration: None,
                        prompt_eval_count: None,
                        eval_count: None,
                    };
                    let mut line = serde_json::to_vec(&frame).unwrap_or_default();
                    line.push(b'\n');
                    let _ = tx.send(Ok(Bytes::from(line))).await;
                    continue;
                }
                let Some(data) = line.strip_prefix("data: ") else { continue };
                let Ok(parsed) = serde_json::from_str::<ChatCompletionResponse>(data) else { continue };

                let content = parsed
                    .choices
                    .first()
                    .and_then(|c| c.delta.as_ref().map(|d| d.content.clone()))
                    .unwrap_or_default();

                let frame = OllamaStreamFrame {
                    model: model.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    response: if is_chat { None } else { Some(content.clone()) },
                    message: if is_chat {
                        Some(ChatMessage {
                            role: "assistant".into(),
                            content,
                        })
                    } else {
                        None
                    },
                    done: false,
                    total_duration: None,
                    prompt_eval_count: parsed.usage.as_ref().map(|u| u.prompt_tokens),
                    eval_count: parsed.usage.as_ref().map(|u| u.completion_tokens),
                };
                let mut line = serde_json::to_vec(&frame).unwrap_or_default();
                line.push(b'\n');
                if tx.send(Ok(Bytes::from(line))).await.is_err() {
                    return;
                }
            }
        }
    });

    Response::builder()
        .status(200)
        .header("Content-Type", "application/x-ndjson")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .map_err(|e| ApiError::internal(e.to_string()))
}

/// Non-streaming Ollama proxy: collect the upstream response in full, then
/// emit it as a single NDJSON frame with `done: true`. Used when the client
/// passes `stream: false`.
async fn proxy_to_ollama_single(
    state: Arc<AppState>,
    upstream_url: String,
    body: Bytes,
    model: String,
    is_chat: bool,
) -> Result<Response, ApiError> {
    let resp = state
        .http_client
        .post(&upstream_url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| ApiError::internal(format!("Upstream failed: {}", e)))?;

    let parsed: ChatCompletionResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::internal(format!("Upstream parse failed: {}", e)))?;

    let content = parsed
        .choices
        .first()
        .and_then(|c| {
            c.message
                .as_ref()
                .map(|m| m.content.clone())
                .or_else(|| c.delta.as_ref().map(|d| d.content.clone()))
        })
        .unwrap_or_default();

    let frame = OllamaStreamFrame {
        model,
        created_at: chrono::Utc::now().to_rfc3339(),
        response: if is_chat { None } else { Some(content.clone()) },
        message: if is_chat {
            Some(ChatMessage {
                role: "assistant".into(),
                content,
            })
        } else {
            None
        },
        done: true,
        total_duration: None,
        prompt_eval_count: parsed.usage.as_ref().map(|u| u.prompt_tokens),
        eval_count: parsed.usage.as_ref().map(|u| u.completion_tokens),
    };

    Ok(Json(frame).into_response())
}

// =============================================================================
// /api/disk-usage and /api/gc
// =============================================================================

/// `GET /api/disk-usage` — per-model byte usage plus an "orphans" list of
/// files in models_dir/gguf_dir that aren't referenced by any manifest.
/// Useful for figuring out where disk went after lots of pulls and rms.
async fn disk_usage_handler(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let manifests = state.registry.list();
    let mut models = Vec::new();
    let mut total: u64 = 0;
    let mut referenced_paths: std::collections::HashSet<std::path::PathBuf> = Default::default();

    for m in &manifests {
        let local_bytes = dir_size(&m.local_path).await;
        let gguf_bytes = match &m.gguf_path {
            Some(p) => file_size(p).await,
            None => 0,
        };
        if m.local_path.exists() {
            referenced_paths.insert(m.local_path.clone());
        }
        if let Some(p) = &m.gguf_path {
            if p.exists() {
                referenced_paths.insert(p.clone());
            }
        }
        total += local_bytes + gguf_bytes;
        models.push(DiskUsageEntry {
            alias: m.alias.clone(),
            local_path_bytes: local_bytes,
            gguf_bytes,
        });
    }

    // Find orphans: top-level entries under models_dir and gguf_dir not referenced
    let (orphans, orphan_bytes) =
        find_orphans(&state.settings.models_dir, &state.settings.gguf_dir, &referenced_paths)
            .await;

    Ok(Json(DiskUsageResponse {
        models,
        total_bytes: total,
        orphans,
        orphan_bytes,
    }))
}

/// `POST /api/gc` — delete the orphan files surfaced by `/api/disk-usage`.
/// Returns lists of `removed` and `errors`. Destructive — no confirmation,
/// callers should preview with `/api/disk-usage` first.
async fn gc_handler(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let manifests = state.registry.list();
    let mut referenced: std::collections::HashSet<std::path::PathBuf> = Default::default();
    for m in &manifests {
        if m.local_path.exists() {
            referenced.insert(m.local_path.clone());
        }
        if let Some(p) = &m.gguf_path {
            if p.exists() {
                referenced.insert(p.clone());
            }
        }
    }

    let (orphans, _) =
        find_orphans(&state.settings.models_dir, &state.settings.gguf_dir, &referenced).await;
    let mut removed = Vec::new();
    let mut errors = Vec::new();

    for orphan in &orphans {
        let path = std::path::PathBuf::from(orphan);
        let result = if path.is_dir() {
            tokio::fs::remove_dir_all(&path).await
        } else {
            tokio::fs::remove_file(&path).await
        };
        match result {
            Ok(_) => removed.push(orphan.clone()),
            Err(e) => errors.push(format!("{}: {}", orphan, e)),
        }
    }

    Ok(Json(serde_json::json!({
        "removed": removed,
        "errors": errors,
    })))
}

/// Walk both data directories and surface any top-level entry whose path isn't
/// in the `referenced` set (i.e. not pointed at by any manifest's local_path
/// or gguf_path). Returns the orphan paths and their total size in bytes.
async fn find_orphans(
    models_dir: &std::path::Path,
    gguf_dir: &std::path::Path,
    referenced: &std::collections::HashSet<std::path::PathBuf>,
) -> (Vec<String>, u64) {
    let mut orphans = Vec::new();
    let mut total: u64 = 0;
    for dir in [models_dir, gguf_dir] {
        let Ok(mut rd) = tokio::fs::read_dir(dir).await else { continue };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let path = entry.path();
            if !referenced.contains(&path) {
                let size = if path.is_dir() {
                    dir_size(&path).await
                } else {
                    file_size(&path).await
                };
                total += size;
                orphans.push(path.to_string_lossy().to_string());
            }
        }
    }
    (orphans, total)
}

/// File size in bytes, or 0 if the file is missing / unreadable. Used by
/// disk-usage and orphan-discovery; treats errors as "not present" since
/// we don't want a transient stat failure to crash the endpoint.
async fn file_size(p: &std::path::Path) -> u64 {
    tokio::fs::metadata(p).await.map(|m| m.len()).unwrap_or(0)
}

/// Recursively sum the size of every file under `root`. Uses an explicit
/// stack instead of async recursion (async fns can't recurse without boxing
/// the future, which would allocate per directory).
async fn dir_size(root: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(mut rd) = tokio::fs::read_dir(&dir).await else { continue };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let Ok(meta) = entry.metadata().await else { continue };
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                stack.push(entry.path());
            }
        }
    }
    total
}

// =============================================================================
// Token counting helpers
// =============================================================================

/// Parse one SSE line and, if it carries a chat-completion event, attribute the
/// `delta.content` text to the `tokens_generated_total` counter. Also captures
/// `usage.prompt_tokens` and `usage.completion_tokens` from the final event.
///
/// We count by whitespace-split words — a fast approximation of true token
/// counts (~10% off for English). Cheaper than running a tokenizer and good
/// enough for throughput trending.
fn count_tokens_in_sse_line(line: &[u8], metrics: &Metrics) {
    // SSE format is `data: <json>\n` (and `data: [DONE]\n`). Strip the prefix
    // and trim CR if present.
    let line = match std::str::from_utf8(line) {
        Ok(s) => s.trim_end_matches('\r'),
        Err(_) => return,
    };
    let Some(payload) = line.strip_prefix("data: ") else { return };
    if payload == "[DONE]" {
        return;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else { return };

    // delta.content (streaming chunks)
    if let Some(content) = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|s| s.as_str())
    {
        if !content.is_empty() {
            let approx = content.split_whitespace().count().max(1) as u64;
            metrics
                .tokens_generated_total
                .fetch_add(approx, Ordering::Relaxed);
        }
    }

    // Final event with usage block — preferred when present (exact counts)
    if let Some(usage) = v.get("usage") {
        if let Some(p) = usage.get("prompt_tokens").and_then(|x| x.as_u64()) {
            metrics.prompt_tokens_total.fetch_add(p, Ordering::Relaxed);
        }
        if let Some(c) = usage.get("completion_tokens").and_then(|x| x.as_u64()) {
            // Replace approximate count with exact one from upstream.
            metrics
                .tokens_generated_total
                .fetch_add(c, Ordering::Relaxed);
        }
    }
}
