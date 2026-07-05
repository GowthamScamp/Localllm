//! # Model registry
//!
//! In-memory `ManifestStore` keyed by alias, backed by one JSON file per model
//! under `manifests_dir`. Loaded lazily at daemon startup, kept in sync on
//! every save/delete. Reads are `Arc<ModelManifest>` clones (cheap pointer
//! bumps), writes are atomic via temp-file + rename.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// On-disk storage format of model weights.
// GGUF is an established file-format name, not an accidental SHOUT; allow it.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WeightFormat {
    /// HuggingFace's standard tensor format. Original-precision downloads.
    Safetensors,
    /// llama.cpp's quantized format. Output of the quantization pipeline.
    GGUF,
    /// Legacy PyTorch `.bin`. We filter these out at download time but the
    /// variant exists for manifests created externally.
    PyTorch,
}

/// Quantization levels supported by llama-quantize, smallest to largest.
/// Naming uses Rust-identifier-friendly variants (`Q4KM` not `Q4_K_M`);
/// `to_llama_str` converts back to the canonical llama.cpp name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationLevel {
    Q4_0,
    Q4KM,
    Q5KM,
    Q6K,
    Q8_0,
    F16,
    F32,
}

impl QuantizationLevel {
    /// Canonical llama.cpp name (e.g. `Q4KM` → `"Q4_K_M"`).
    /// Used when shelling out to `llama-quantize` and in API responses.
    pub fn to_llama_str(&self) -> &'static str {
        match self {
            QuantizationLevel::Q4_0 => "Q4_0",
            QuantizationLevel::Q4KM => "Q4_K_M",
            QuantizationLevel::Q5KM => "Q5_K_M",
            QuantizationLevel::Q6K => "Q6_K",
            QuantizationLevel::Q8_0 => "Q8_0",
            QuantizationLevel::F16 => "F16",
            QuantizationLevel::F32 => "F32",
        }
    }

    /// Bytes consumed per model parameter at this precision. Used by VRAM
    /// estimation to predict whether a model will fit on the GPU.
    /// E.g. Q4_K_M ≈ 0.5 B/param (4 bits + small metadata), F16 = 2 B/param.
    pub fn bytes_per_param(&self) -> f32 {
        match self {
            QuantizationLevel::Q4_0 => 0.5,
            QuantizationLevel::Q4KM => 0.5,
            QuantizationLevel::Q5KM => 0.625,
            QuantizationLevel::Q6K => 0.75,
            QuantizationLevel::Q8_0 => 1.0,
            QuantizationLevel::F16 => 2.0,
            QuantizationLevel::F32 => 4.0,
        }
    }

    /// Parse a user-supplied quantization name. Accepts both styles
    /// (`"Q4_K_M"` and `"Q4KM"`); case-insensitive. Returns `None` for
    /// unknown levels — caller should surface as a `bad_request` API error.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "Q4_0" => Some(QuantizationLevel::Q4_0),
            "Q4_K_M" | "Q4KM" => Some(QuantizationLevel::Q4KM),
            "Q5_K_M" | "Q5KM" => Some(QuantizationLevel::Q5KM),
            "Q6_K" | "Q6K" => Some(QuantizationLevel::Q6K),
            "Q8_0" => Some(QuantizationLevel::Q8_0),
            "F16" => Some(QuantizationLevel::F16),
            "F32" => Some(QuantizationLevel::F32),
            _ => None,
        }
    }

    /// Infer the quant level from a GGUF filename, e.g.
    /// `"Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"` → `Q4KM`. Scans for each known
    /// level's canonical token (case-insensitive). Returns `None` if no token
    /// matches. Checks longer/more-specific tokens first so `Q4_K_M` isn't
    /// shadowed by a substring match on `Q4_0`.
    pub fn from_filename(name: &str) -> Option<Self> {
        let n = name.to_uppercase();
        // Order matters: most specific first.
        let levels = [
            QuantizationLevel::Q4KM,
            QuantizationLevel::Q5KM,
            QuantizationLevel::Q6K,
            QuantizationLevel::Q8_0,
            QuantizationLevel::Q4_0,
            QuantizationLevel::F16,
            QuantizationLevel::F32,
        ];
        levels
            .into_iter()
            .find(|l| n.contains(l.to_llama_str()))
    }
}

/// One file inside a model's local directory. `sha256` may be empty if the
/// download path didn't verify (current HF pulls skip per-file hashes since
/// HF doesn't return them in metadata — we'd have to recompute post-download).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Everything we know about a registered model. One per `<alias>.json` file
/// on disk; serialized via serde. The `#[serde(default)]` on newer fields
/// (`embeddings`, `modelfile`) means old manifest files still deserialize
/// cleanly when we add new fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub repo_id: String,
    pub alias: String,
    pub revision: String,
    pub local_path: PathBuf,
    pub architecture: String,
    pub weight_format: WeightFormat,
    pub parameters_billion: f32,
    pub context_length: u32,
    pub quantization: Option<QuantizationLevel>,
    pub gguf_path: Option<PathBuf>,
    pub files: Vec<FileEntry>,
    pub downloaded_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    /// Whether to start the inference backend with the embeddings flag.
    #[serde(default)]
    pub embeddings: bool,
    /// Optional Modelfile (SYSTEM prompt, TEMPLATE, PARAMETERS).
    /// Stored alongside the manifest so /api/show and /api/chat can use it.
    #[serde(default)]
    pub modelfile: Option<crate::registry::modelfile::Modelfile>,
}

/// Thread-safe alias→manifest cache backed by JSON files on disk.
///
/// The mutex protects the HashMap structure only; values are `Arc`-wrapped so
/// the lock is released before any callback ever touches a manifest. This means
/// reads/writes contend only briefly, and a slow handler never blocks the
/// registry for everyone.
pub struct ManifestStore {
    manifests_dir: PathBuf,
    cache: Mutex<HashMap<String, std::sync::Arc<ModelManifest>>>,
    /// A5 — set of aliases whose `last_used` has drifted from what's persisted.
    /// Drained by `flush_dirty()` which folds `last_used_live` into the manifest.
    dirty: Mutex<std::collections::HashSet<String>>,
    /// OPT-3 — live `last_used` timestamps (Unix seconds) updated on the request
    /// hot path. Writing here is a single map insert with no `ModelManifest`
    /// clone — far cheaper than cloning the whole manifest just to bump a time.
    /// Folded into the persisted manifest by `flush_dirty()` every 30s.
    last_used_live: Mutex<HashMap<String, i64>>,
}

/// Recover from a poisoned mutex with a warning instead of a panic.
///
/// A poisoned mutex means another task panicked while holding it. The data
/// inside is still consistent (no torn writes), but Rust's API forces us to
/// acknowledge the panic. We log it and continue — losing the registry over a
/// crash in unrelated code would be a bad UX trade.
fn lock_or_recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p: PoisonError<MutexGuard<'_, T>>| {
        tracing::warn!("Recovering from poisoned mutex (a task panicked while holding it)");
        p.into_inner()
    })
}

impl ManifestStore {
    /// Build an empty registry rooted at `manifests_dir`. Creates the
    /// directory if it doesn't exist. Does NOT scan disk — call `load_all`
    /// for that (separately so the daemon can do it in a background task and
    /// start serving requests immediately).
    pub fn new(manifests_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&manifests_dir)?;
        Ok(Self {
            manifests_dir,
            cache: Mutex::new(HashMap::new()),
            dirty: Mutex::new(std::collections::HashSet::new()),
            last_used_live: Mutex::new(HashMap::new()),
        })
    }

    /// Scan `manifests_dir` for `*.json` files, parse each, insert into the
    /// cache. Tolerant: malformed files are logged and skipped (they shouldn't
    /// take the daemon down). Uses synchronous I/O — intended to be called via
    /// `tokio::task::spawn_blocking` from async contexts.
    pub fn load_all(&self) {
        let dir = &self.manifests_dir;
        if !dir.exists() {
            return;
        }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("Failed to scan manifests dir {:?}: {}", dir, e);
                return;
            }
        };

        let mut loaded = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(content) => match serde_json::from_str::<ModelManifest>(&content) {
                    Ok(manifest) => {
                        let alias = manifest.alias.clone();
                        let mut cache = lock_or_recover(&self.cache);
                        cache.insert(alias, std::sync::Arc::new(manifest));
                        loaded += 1;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse manifest {:?}: {}", path, e);
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read manifest {:?}: {}", path, e);
                }
            }
        }
        tracing::info!("Loaded {} manifests from {:?}", loaded, dir);
    }

    /// Compute the on-disk path for a manifest. Slashes / backslashes / colons
    /// in the alias are replaced with `_` — protects against an alias like
    /// `"foo/../bar"` writing outside `manifests_dir`.
    fn manifest_path(&self, alias: &str) -> PathBuf {
        let safe_alias = alias.replace(['/', '\\', ':'], "_");
        self.manifests_dir.join(format!("{}.json", safe_alias))
    }

    /// Atomically persist a manifest. Writes to a UUID-suffixed temp file
    /// first, then renames into place — so a crash mid-write can't corrupt
    /// an existing manifest, and two concurrent saves can't collide on the
    /// temp file name. Updates the in-memory cache after the rename succeeds.
    pub fn save(&self, manifest: &ModelManifest) -> Result<()> {
        let path = self.manifest_path(&manifest.alias);
        // UUID-suffixed temp file so concurrent saves can't collide.
        let tmp_path = self.manifests_dir.join(format!(
            "{}.{}.tmp",
            path.file_stem().unwrap_or_default().to_string_lossy(),
            uuid::Uuid::new_v4()
        ));
        let content = serde_json::to_string_pretty(manifest)?;
        std::fs::write(&tmp_path, &content)?;
        std::fs::rename(&tmp_path, &path)?;

        let mut cache = lock_or_recover(&self.cache);
        cache.insert(manifest.alias.clone(), std::sync::Arc::new(manifest.clone()));
        Ok(())
    }

    /// Remove a manifest's JSON file and drop it from the cache.
    /// Does NOT delete the model weights — that's handled by the
    /// `delete_model_inner` API handler.
    pub fn delete(&self, alias: &str) -> Result<()> {
        let path = self.manifest_path(alias);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        lock_or_recover(&self.cache).remove(alias);
        // Drop any pending hot-path state so it can't leak or resurrect a
        // deleted alias on the next flush.
        lock_or_recover(&self.dirty).remove(alias);
        lock_or_recover(&self.last_used_live).remove(alias);
        Ok(())
    }

    /// Look up by exact alias. Returns `Arc<ModelManifest>` so callers can
    /// drop the registry lock immediately and hold the manifest as long as
    /// they need without contention.
    pub fn get(&self, alias: &str) -> Option<std::sync::Arc<ModelManifest>> {
        let cache = lock_or_recover(&self.cache);
        cache.get(alias).cloned()
    }

    /// Snapshot of every registered manifest. Order is undefined (HashMap).
    /// Only the `Vec` is allocated — the manifests themselves are shared `Arc`s.
    pub fn list(&self) -> Vec<std::sync::Arc<ModelManifest>> {
        let cache = lock_or_recover(&self.cache);
        cache.values().cloned().collect()
    }

    /// Bump a model's `last_used` timestamp to now and persist.
    /// Currently unused (we touch `last_used` on the inference-process atomic
    /// instead), but kept for future "warm-set" eviction strategies that need
    /// to know recency at the manifest level.
    /// A5 — In-memory-only timestamp touch. Marks the alias as "dirty" so the
    /// daemon's periodic flush task persists it to disk later. Avoids the
    /// per-request disk write of the original `save()` path.
    /// OPT-3 — Hot-path LRU bump. Records the current time in `last_used_live`
    /// and marks the alias dirty. **No `ModelManifest` clone** — that only
    /// happens once per flush interval in `flush_dirty`. Cheap enough to call
    /// on every inference request.
    pub fn update_last_used(&self, alias: &str) -> Result<()> {
        // Only track aliases we actually know about, to avoid unbounded growth
        // from bogus requests. A read-lock check is cheap.
        {
            let cache = lock_or_recover(&self.cache);
            if !cache.contains_key(alias) {
                return Err(anyhow::anyhow!("Manifest not found: {}", alias));
            }
        }
        let now = Utc::now().timestamp();
        lock_or_recover(&self.last_used_live).insert(alias.to_string(), now);
        lock_or_recover(&self.dirty).insert(alias.to_string());
        Ok(())
    }

    /// Write all dirty manifests to disk and clear the dirty set. Called by the
    /// daemon's periodic flush task (every 30s) and on graceful shutdown.
    pub fn flush_dirty(&self) {
        // Collect dirty aliases while holding the lock briefly, then release.
        let dirty_now: Vec<String> = {
            let mut dirty = lock_or_recover(&self.dirty);
            let v: Vec<_> = dirty.iter().cloned().collect();
            dirty.clear();
            v
        };
        for alias in dirty_now {
            if let Some(manifest) = self.get(&alias) {
                // Fold the live timestamp (if any) into the manifest before
                // persisting. This is the single clone per flushed alias — once
                // every 30s, not once per request. `save()` refreshes the cache.
                let live_ts = lock_or_recover(&self.last_used_live).get(&alias).copied();
                let mut to_save = (*manifest).clone();
                if let Some(ts) = live_ts {
                    if let Some(dt) = chrono::DateTime::<Utc>::from_timestamp(ts, 0) {
                        to_save.last_used = dt;
                    }
                }
                if let Err(e) = self.save(&to_save) {
                    tracing::warn!("Failed to flush manifest {}: {}", alias, e);
                    // Re-mark as dirty so we try again next interval.
                    lock_or_recover(&self.dirty).insert(alias);
                }
            }
        }
    }

    /// Look up by HuggingFace repo_id (exact match OR substring). The substring
    /// fallback is intentional — lets users pass short names like
    /// `"Llama-3.2-1B"` and still hit `"meta-llama/Llama-3.2-1B-Instruct"`.
    /// First match wins; ambiguous prefixes are a known UX hazard.
    pub fn find_by_repo_id(&self, repo_id: &str) -> Option<std::sync::Arc<ModelManifest>> {
        let cache = lock_or_recover(&self.cache);
        cache
            .values()
            .find(|m| m.repo_id == repo_id || m.repo_id.contains(repo_id))
            .cloned()
    }

    /// Create a new manifest pointing at the same on-disk weights as an
    /// existing one. Used by Ollama-style `cp` and `create` to alias a model
    /// (or to fork it with different Modelfile overrides). Fails if the
    /// destination alias is already taken.
    pub fn clone_as(&self, src_alias: &str, dst_alias: &str) -> Result<std::sync::Arc<ModelManifest>> {
        let src = self
            .get(src_alias)
            .ok_or_else(|| anyhow::anyhow!("Source model not found: {}", src_alias))?;
        if self.get(dst_alias).is_some() {
            return Err(anyhow::anyhow!("Destination alias already exists: {}", dst_alias));
        }
        let mut cloned = (*src).clone();
        cloned.alias = dst_alias.to_string();
        cloned.downloaded_at = Utc::now();
        cloned.last_used = Utc::now();
        self.save(&cloned)?;
        Ok(self
            .get(dst_alias)
            .expect("just saved, must be present"))
    }

    pub fn manifests_dir(&self) -> &Path {
        &self.manifests_dir
    }
}

impl std::fmt::Debug for ManifestStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManifestStore")
            .field("manifests_dir", &self.manifests_dir)
            .finish()
    }
}

/// Derive a user-friendly alias from a HF repo_id.
/// E.g. `"meta-llama/Llama-3.2-1B-Instruct"` → `"llama-3.2-1b-instruct"`.
/// Drops the org prefix, lowercases, replaces dots/spaces with hyphens.
pub fn alias_from_repo_id(repo_id: &str) -> String {
    repo_id
        .split('/')
        .next_back()
        .unwrap_or(repo_id)
        .to_lowercase()
        .replace(['.', ' '], "-")
}

/// Convert a repo_id into a filesystem-safe directory name.
/// `"org/model"` → `"org--model"`. Inverse op isn't needed; we never need
/// to recover the original from the sanitized form.
pub fn sanitize_path(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// Reject filenames that could escape the model directory if blindly joined
/// to a parent path. Defense against a compromised or malicious HF mirror
/// returning entries like `"../../etc/passwd"` or `"C:\Windows\..."`.
///
/// Rules:
///   * Non-empty.
///   * No NUL or ASCII control chars (which can confuse path parsing).
///   * No `..` traversal sequence.
///   * No absolute-path prefix (`/`, `\`, or `X:` Windows drive).
pub fn is_safe_filename(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    // Reject NUL and ASCII control chars
    if name.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return false;
    }
    // Reject path traversal
    if name.contains("..") || name.starts_with('/') || name.starts_with('\\') {
        return false;
    }
    // Reject Windows drive prefix
    if name.len() >= 2 && name.as_bytes()[1] == b':' {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_level_roundtrip() {
        for s in &["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32"] {
            let q = QuantizationLevel::from_str(s).expect("known level");
            assert_eq!(q.to_llama_str(), *s);
        }
    }

    #[test]
    fn quantization_level_short_form_accepted() {
        assert_eq!(
            QuantizationLevel::from_str("Q4KM"),
            Some(QuantizationLevel::Q4KM)
        );
        assert_eq!(
            QuantizationLevel::from_str("q5_k_m"),
            Some(QuantizationLevel::Q5KM)
        );
    }

    #[test]
    fn quantization_level_unknown_returns_none() {
        assert_eq!(QuantizationLevel::from_str("Q9_X_Z"), None);
        assert_eq!(QuantizationLevel::from_str(""), None);
    }

    #[test]
    fn quantization_level_from_filename() {
        assert_eq!(
            QuantizationLevel::from_filename("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
            Some(QuantizationLevel::Q4KM)
        );
        assert_eq!(
            QuantizationLevel::from_filename("model-q8_0.gguf"),
            Some(QuantizationLevel::Q8_0)
        );
        assert_eq!(
            QuantizationLevel::from_filename("model-Q6_K.gguf"),
            Some(QuantizationLevel::Q6K)
        );
        // No recognizable quant token.
        assert_eq!(QuantizationLevel::from_filename("model.gguf"), None);
    }

    #[test]
    fn bytes_per_param_monotonic() {
        // Higher precision must use more bytes per parameter.
        assert!(QuantizationLevel::Q4_0.bytes_per_param() < QuantizationLevel::Q5KM.bytes_per_param());
        assert!(QuantizationLevel::Q5KM.bytes_per_param() < QuantizationLevel::Q8_0.bytes_per_param());
        assert!(QuantizationLevel::Q8_0.bytes_per_param() < QuantizationLevel::F16.bytes_per_param());
        assert!(QuantizationLevel::F16.bytes_per_param() < QuantizationLevel::F32.bytes_per_param());
    }

    #[test]
    fn safe_filename_accepts_normal_files() {
        assert!(is_safe_filename("model.safetensors"));
        assert!(is_safe_filename("config.json"));
        assert!(is_safe_filename("model-00001-of-00002.safetensors"));
        assert!(is_safe_filename("tokenizer.model"));
    }

    #[test]
    fn safe_filename_rejects_traversal() {
        assert!(!is_safe_filename("../etc/passwd"));
        assert!(!is_safe_filename("..\\windows\\system32"));
        assert!(!is_safe_filename("/etc/passwd"));
        assert!(!is_safe_filename("\\windows\\system32"));
        assert!(!is_safe_filename("C:/Windows/System32/cmd.exe"));
        assert!(!is_safe_filename("subdir/../escape"));
    }

    #[test]
    fn safe_filename_rejects_control_chars() {
        assert!(!is_safe_filename(""));
        assert!(!is_safe_filename("foo\0bar"));
        assert!(!is_safe_filename("with\nnewline"));
        assert!(!is_safe_filename("with\ttab"));
    }

    #[test]
    fn alias_from_repo_id_lowercases_and_dashes() {
        assert_eq!(alias_from_repo_id("meta-llama/Llama-3.2-1B"), "llama-3-2-1b");
    }

    #[test]
    fn sanitize_path_replaces_slashes() {
        assert_eq!(
            sanitize_path("meta-llama/Llama-3.2-1B"),
            "meta-llama--Llama-3.2-1B"
        );
    }
}
