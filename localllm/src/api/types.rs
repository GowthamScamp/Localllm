//! # API wire types
//!
//! Three families of request/response shapes:
//!
//! 1. **OpenAI-compatible** — `/v1/*` endpoints. Drop-in for any OpenAI SDK.
//! 2. **Ollama-compatible** — `/api/*` endpoints. Drop-in for Ollama clients.
//! 3. **Internal/native** — `/api/pull`, `/api/quantize`, `/api/load` — used
//!    by the localllm CLI itself.
//!
//! All types derive `Serialize + Deserialize` because the same struct is used
//! on both daemon (response) and CLI (request) sides — no duplication.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

// =============================================================================
// OpenAI-compatible types
// =============================================================================

/// A single chat turn. `role` is one of `system | user | assistant`.
/// Same shape on input (request) and output (streaming delta).
///
/// `content` deserializes a JSON `null` or a missing field as an empty string.
/// This matters for streaming deltas: the *first* SSE chunk from llama.cpp sends
/// `{"role":"assistant","content":null}` (role announcement, no text yet). With a
/// plain `String` field that chunk fails to parse and the whole reply is dropped.
/// `role` gets the same treatment for symmetry (some backends omit it on deltas).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    #[serde(default, deserialize_with = "string_or_null")]
    pub role: String,
    #[serde(default, deserialize_with = "string_or_null")]
    pub content: String,
}

/// Deserialize a JSON string, treating `null` (and absence, via `#[serde(default)]`)
/// as an empty string. Rejects non-string, non-null values.
fn string_or_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// Request body for `POST /v1/chat/completions`. Only `model` and `messages`
/// are required; the rest are inference hyperparameters that pass through to
/// the upstream backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
}

/// One element of `choices[]`. Either `message` (non-streaming) or `delta`
/// (streaming) is populated — never both. `finish_reason` is only on the final
/// chunk: `stop | length | content_filter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Option<ChatMessage>,
    pub delta: Option<ChatMessage>,
    pub finish_reason: Option<String>,
}

/// Token counts returned in the final SSE event (or in non-streaming responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Shape of every SSE event for chat completions, AND the full response body
/// for non-streaming requests. `usage` is only on the final chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
}

/// One row in the OpenAI `GET /v1/models` response. `object` is always `"model"`.
/// `created` is a Unix timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

impl ModelObject {
    /// Build a `ModelObject` from the alias + download timestamp. `owned_by`
    /// is hardcoded to `"localllm"` — OpenAI uses this to indicate provider.
    pub fn new(id: String, created: i64) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created,
            owned_by: "localllm".to_string(),
        }
    }
}

/// Top-level wrapper for `GET /v1/models`. OpenAI returns
/// `{"object":"list", "data":[...]}` — the `object` field is always `"list"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

impl ModelListResponse {
    pub fn new(data: Vec<ModelObject>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

// =============================================================================
// OpenAI Embeddings
// =============================================================================

/// `input` may be a single string or a list of strings. Use `serde_json::Value` so
/// we can pass it through to the upstream verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Option<Usage>,
}

// =============================================================================
// Internal / CLI types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub repo_id: String,
    pub revision: Option<String>,
    pub quantize: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Optional HuggingFace access token for this pull. Lets the CLI forward a
    /// `--hf-token` (or `HF_TOKEN`) so gated/private models download — and
    /// authenticated requests get higher HF rate limits, so public pulls are
    /// faster too. When absent, the daemon falls back to its own `HF_TOKEN`.
    #[serde(default)]
    pub hf_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeRequest {
    pub alias: String,
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsEntry {
    pub alias: String,
    pub port: u16,
    pub backend: String,
    pub started_at: String,
    pub last_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusEntry {
    pub alias: String,
    pub repo_id: String,
    pub architecture: String,
    pub parameters_billion: f32,
    pub quantization: Option<String>,
    pub weight_format: String,
    pub status: String,
    /// Absolute path to the GGUF file if one exists on disk, else null. Lets the
    /// CLI tell "already has a runnable GGUF" from "needs quantizing" without a
    /// second round-trip.
    pub gguf_path: Option<String>,
}

// =============================================================================
// Ollama-compatible types
// =============================================================================

/// GET /api/tags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// POST /api/show
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaShowRequest {
    pub name: String,
    #[serde(default)]
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: OllamaModelDetails,
    pub model_info: serde_json::Value,
}

/// POST /api/generate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default)]
    pub options: serde_json::Value,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
}

/// POST /api/chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default)]
    pub options: serde_json::Value,
}

fn default_stream() -> bool {
    true
}

/// One frame of an Ollama NDJSON streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaStreamFrame {
    pub model: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>, // /api/generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<ChatMessage>, // /api/chat
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

/// POST /api/embeddings (Ollama-style — singular `input` typo preserved as `prompt`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingsRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub options: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

/// POST /api/copy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaCopyRequest {
    pub source: String,
    pub destination: String,
}

/// POST /api/create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaCreateRequest {
    pub name: String,
    pub modelfile: String,
    #[serde(default = "default_stream")]
    pub stream: bool,
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    /// User-facing hint for remediation, e.g. "Run: localllm pull <repo>".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

impl ApiError {
    /// Generic constructor. Prefer `internal` / `not_found` / `bad_request`
    /// for the common cases — `error_type` controls the HTTP status code.
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ApiErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                hint: None,
            },
        }
    }

    /// Attach a user-facing remediation hint (e.g. `"Run: localllm pull <repo>"`).
    /// Chainable: `ApiError::not_found(msg).with_hint("...")`.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.error.hint = Some(hint.into());
        self
    }

    /// 500 Internal Server Error — backend failure, panic recovery, etc.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(message, "internal_error")
    }

    /// 404 Not Found — typically used when a model alias isn't in the registry.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(message, "not_found")
    }

    /// 400 Bad Request — malformed JSON, missing fields, unknown enum value.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(message, "invalid_request_error")
    }
}

impl IntoResponse for ApiError {
    /// Map the symbolic `error_type` to an HTTP status code and serialize the
    /// error body as JSON. Axum calls this automatically when a handler returns
    /// `Result<_, ApiError>` and the result is `Err`.
    fn into_response(self) -> Response {
        let status = match self.error.error_type.as_str() {
            "not_found" => StatusCode::NOT_FOUND,
            "invalid_request_error" => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self)).into_response()
    }
}

// =============================================================================
// Disk usage
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsageEntry {
    pub alias: String,
    pub local_path_bytes: u64,
    pub gguf_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsageResponse {
    pub models: Vec<DiskUsageEntry>,
    pub total_bytes: u64,
    /// Files in models_dir or gguf_dir that aren't referenced by any manifest.
    pub orphans: Vec<String>,
    pub orphan_bytes: u64,
}
