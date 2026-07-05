//! # Download orchestration
//!
//! `DownloadManager::pull` is the top-level "give me a model" operation.
//! It picks up where `HuggingFaceClient` leaves off: list files, spawn
//! parallel downloads (gated by a semaphore), parse config.json to derive
//! metadata, and return a fully-populated `ModelManifest` ready for the
//! registry.

use anyhow::{anyhow, Result};
use chrono::Utc;
use futures::future::join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::config::Settings;
use crate::downloader::hf_api::{HuggingFaceClient, RepoFile};
use crate::registry::manifest::{
    alias_from_repo_id, is_safe_filename, sanitize_path, FileEntry, ModelManifest,
    QuantizationLevel, WeightFormat,
};

/// Progress event emitted during a GGUF pull, forwarded to the CLI as NDJSON.
#[derive(Debug, Clone)]
pub enum PullProgress {
    /// A human-readable phase label (e.g. "resolving GGUF source").
    Status(String),
    /// Byte-level download progress: (downloaded, total). total=0 if unknown.
    Download { downloaded: u64, total: u64 },
}

pub struct DownloadManager {
    pub hf_client: Arc<HuggingFaceClient>,
    pub settings: Arc<Settings>,
}

impl DownloadManager {
    pub fn new(hf_client: Arc<HuggingFaceClient>, settings: Arc<Settings>) -> Self {
        Self {
            hf_client,
            settings,
        }
    }

    /// Download an entire model. Steps:
    ///   1. Resolve the commit hash (caller's override OR HF "main" pointer).
    ///   2. Ensure the local destination dir exists under `models_dir/<safe_repo>/<rev>`.
    ///   3. Get the filtered file list (config + tokenizer + safetensors only).
    ///   4. Pre-validate every filename (path-traversal guard) — abort the whole
    ///      pull if any single name is unsafe.
    ///   5. Canonicalize the local dir so per-file containment checks can verify
    ///      the resolved path doesn't escape it after symlink resolution.
    ///   6. Spawn one tokio task per file, gated by `Semaphore` for parallelism
    ///      limit. Each task does its own retry/resume via `download_file`.
    ///   7. Wait for all tasks; fail fast on the first error.
    ///   8. Parse `config.json` to derive architecture, context length, and a
    ///      rough parameter count.
    ///   9. Build and return a `ModelManifest` (caller saves it to the registry).
    pub async fn pull(&self, repo_id: &str, revision: Option<&str>) -> Result<ModelManifest> {
        tracing::info!("Pulling model: {}", repo_id);

        let info = self.hf_client.get_repo_info(repo_id).await?;
        let revision_hash = if let Some(rev) = revision {
            rev.to_string()
        } else {
            info.revision.clone()
        };

        let safe_repo = sanitize_path(repo_id);
        let local_dir = self
            .settings
            .models_dir
            .join(&safe_repo)
            .join(&revision_hash);

        tokio::fs::create_dir_all(&local_dir).await?;

        let files = self
            .hf_client
            .list_files(repo_id, &revision_hash)
            .await?;

        if files.is_empty() {
            return Err(anyhow!(
                "No suitable files found in repository {}",
                repo_id
            ));
        }

        let multi = Arc::new(MultiProgress::new());
        let semaphore = Arc::new(Semaphore::new(self.settings.max_concurrent_downloads));

        let style = ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-");

        // Validate filenames *before* spawning download tasks. Any traversal
        // attempt aborts the entire pull immediately.
        for f in &files {
            if !is_safe_filename(&f.rfilename) {
                return Err(anyhow!(
                    "Refusing to download unsafe filename from HF: '{}'",
                    f.rfilename
                ));
            }
        }

        // C2 — canonicalize local_dir so we can verify each dest path stays inside it.
        let local_dir_canonical = tokio::fs::canonicalize(&local_dir)
            .await
            .unwrap_or_else(|_| local_dir.clone());

        // Share repo_id and revision via Arc<str> instead of per-file String clone.
        let repo_id_arc: Arc<str> = Arc::from(repo_id);
        let revision_arc: Arc<str> = Arc::from(revision_hash.as_str());

        let mut handles = Vec::new();

        for file in &files {
            let pb = multi.add(ProgressBar::new(file.size.unwrap_or(0)));
            pb.set_style(style.clone());
            pb.set_message(file.rfilename.clone());

            let hf_client = self.hf_client.clone();
            let repo_id_for_task = Arc::clone(&repo_id_arc);
            let revision_for_task = Arc::clone(&revision_arc);
            let filename = file.rfilename.clone();
            let dest = local_dir.join(&filename);

            // Defense in depth: even with is_safe_filename, ensure the joined path
            // doesn't escape local_dir after any symlink resolution.
            if let Some(parent) = dest.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let local_canon_for_check = local_dir_canonical.clone();
            let dest_for_check = dest.clone();

            let sem = semaphore.clone();
            let expected_size = file.size;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await?;

                // Final containment check once the parent dir exists (in case of symlinks)
                if let Ok(canon_dest) = tokio::fs::canonicalize(
                    dest_for_check.parent().unwrap_or(&dest_for_check),
                )
                .await
                {
                    if !canon_dest.starts_with(&local_canon_for_check) {
                        return Err(anyhow!(
                            "Refusing to write outside models dir: {:?}",
                            dest_for_check
                        ));
                    }
                }

                tracing::info!("Downloading: {}", filename);
                hf_client
                    .download_file(
                        &repo_id_for_task,
                        &revision_for_task,
                        &filename,
                        &dest,
                        expected_size,
                        None,
                        &pb,
                    )
                    .await?;
                pb.finish_with_message(format!("Done: {}", filename));
                Ok::<(), anyhow::Error>(())
            });

            handles.push(handle);
        }

        let results = join_all(handles).await;
        for result in results {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow!("Task join error: {}", e)),
            }
        }

        // Parse config.json
        let config_path = local_dir.join("config.json");
        let config_str = tokio::fs::read_to_string(&config_path)
            .await
            .map_err(|e| anyhow!("Failed to read config.json: {}", e))?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

        let architecture = extract_architecture(&config_json);
        let context_length = config_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as u32;

        let parameters_billion = estimate_parameters(&config_json);

        let file_entries: Vec<FileEntry> = files
            .iter()
            .map(|f| FileEntry {
                name: f.rfilename.clone(),
                sha256: String::new(),
                size_bytes: f.size.unwrap_or(0),
            })
            .collect();

        let alias = alias_from_repo_id(repo_id);
        let now = Utc::now();

        let manifest = ModelManifest {
            repo_id: repo_id.to_string(),
            alias,
            revision: revision_hash,
            local_path: local_dir,
            architecture,
            weight_format: WeightFormat::Safetensors,
            parameters_billion,
            context_length,
            quantization: None,
            gguf_path: None,
            files: file_entries,
            downloaded_at: now,
            last_used: now,
            embeddings: false,
            modelfile: None,
        };

        tracing::info!(
            "Successfully pulled model: {} ({:.1}B params)",
            manifest.alias,
            manifest.parameters_billion
        );

        Ok(manifest)
    }

    /// Resolve a GGUF source for `repo_id`. Tries, in order:
    ///   1. The repo itself (it may already ship `.gguf` files).
    ///   2. Well-known community GGUF mirror repos derived from the model name
    ///      (bartowski, unsloth, QuantFactory, etc.).
    ///
    /// Returns `(source_repo_id, gguf_files)` for the first source that has GGUF
    /// files, or `None` if no GGUF source is found anywhere.
    async fn resolve_gguf_source(&self, repo_id: &str) -> Option<(String, Vec<RepoFile>)> {
        // 1. The repo itself.
        if let Ok(files) = self.hf_client.list_gguf_files(repo_id).await {
            if !files.is_empty() {
                return Some((repo_id.to_string(), files));
            }
        }

        // 2. Community mirror repos. Model name is the part after the last '/'.
        let model_name = repo_id.split('/').next_back().unwrap_or(repo_id);
        // Strip a trailing "-GGUF" if the user already passed a GGUF repo name.
        let base = model_name.trim_end_matches("-GGUF").trim_end_matches("-gguf");
        let candidates = [
            format!("bartowski/{base}-GGUF"),
            format!("unsloth/{base}-GGUF"),
            format!("QuantFactory/{base}-GGUF"),
            format!("lmstudio-community/{base}-GGUF"),
        ];
        for cand in candidates {
            if self.hf_client.repo_exists(&cand).await {
                if let Ok(files) = self.hf_client.list_gguf_files(&cand).await {
                    if !files.is_empty() {
                        return Some((cand, files));
                    }
                }
            }
        }
        None
    }

    /// Pull a prebuilt GGUF directly — no torch, no conversion. Steps:
    ///   1. Resolve a GGUF source (repo or community mirror).
    ///   2. Pick the file matching the requested quant level (fallback: smallest).
    ///   3. Download just that one file with byte-level progress callbacks.
    ///   4. Fetch `config.json` from the *original* repo for accurate metadata
    ///      (the GGUF mirror usually has one too; we fall back to it).
    ///   5. Build a GGUF-format manifest with `gguf_path` + `quantization` set.
    ///
    /// `on_progress` receives status + byte-progress events for the CLI bar.
    pub async fn pull_gguf<F>(
        &self,
        repo_id: &str,
        quant: &str,
        mut on_progress: F,
    ) -> Result<ModelManifest>
    where
        F: FnMut(PullProgress),
    {
        on_progress(PullProgress::Status("resolving GGUF source".into()));

        let (source_repo, gguf_files) = self
            .resolve_gguf_source(repo_id)
            .await
            .ok_or_else(|| {
                anyhow!(
                    "No prebuilt GGUF found for '{}' (checked the repo and bartowski/unsloth/QuantFactory mirrors). \
                     Either the model has no GGUF release, or install torch to convert from safetensors.",
                    repo_id
                )
            })?;

        // Pick the file matching the requested quant; fall back to smallest.
        let wanted = quant.to_lowercase().replace('_', "");
        let chosen = gguf_files
            .iter()
            .find(|f| {
                f.rfilename
                    .to_lowercase()
                    .replace('_', "")
                    .contains(&wanted)
            })
            .or_else(|| gguf_files.first())
            .cloned()
            .ok_or_else(|| anyhow!("No GGUF file selectable from {}", source_repo))?;

        let detected_quant = QuantizationLevel::from_filename(&chosen.rfilename)
            .or_else(|| QuantizationLevel::from_str(quant));

        on_progress(PullProgress::Status(format!(
            "downloading {} from {}",
            chosen.rfilename, source_repo
        )));

        // Resolve the source revision for a stable download URL.
        let src_info = self.hf_client.get_repo_info(&source_repo).await?;

        // Destination: gguf_dir/<alias>-<quant>.gguf
        let alias = alias_from_repo_id(repo_id);
        tokio::fs::create_dir_all(&self.settings.gguf_dir).await?;
        let gguf_dest = self.settings.gguf_dir.join(format!(
            "{}-{}.gguf",
            alias,
            detected_quant
                .as_ref()
                .map(|q| q.to_llama_str())
                .unwrap_or("GGUF")
        ));

        self.hf_client
            .download_file_with_callback(
                &source_repo,
                &src_info.revision,
                &chosen.rfilename,
                &gguf_dest,
                chosen.size,
                |downloaded, total| on_progress(PullProgress::Download { downloaded, total }),
            )
            .await?;

        on_progress(PullProgress::Status("reading model metadata".into()));

        // Best-effort metadata from config.json. Try the original repo first,
        // then the GGUF source. If neither has one, fall back to defaults.
        let (architecture, context_length, parameters_billion) =
            match self.fetch_config_metadata(repo_id).await {
                Some(m) => m,
                None => self
                    .fetch_config_metadata(&source_repo)
                    .await
                    .unwrap_or_else(|| ("unknown".to_string(), 4096, 0.0)),
            };

        let now = Utc::now();
        let file_entry = FileEntry {
            name: chosen.rfilename.clone(),
            sha256: String::new(),
            size_bytes: chosen.size.unwrap_or(0),
        };

        let manifest = ModelManifest {
            repo_id: repo_id.to_string(),
            alias,
            revision: src_info.revision,
            // For pure-GGUF pulls there's no safetensors dir; point local_path at
            // the gguf file's parent so disk-usage / cleanup still work.
            local_path: self.settings.gguf_dir.join("__gguf_only__"),
            architecture,
            weight_format: WeightFormat::GGUF,
            parameters_billion,
            context_length,
            quantization: detected_quant,
            gguf_path: Some(gguf_dest),
            files: vec![file_entry],
            downloaded_at: now,
            last_used: now,
            embeddings: false,
            modelfile: None,
        };

        Ok(manifest)
    }

    /// Fetch + parse `config.json` from a repo for metadata. Returns
    /// `(architecture, context_length, parameters_billion)` or `None` if the
    /// repo has no readable config.json (common for GGUF-only mirrors).
    async fn fetch_config_metadata(&self, repo_id: &str) -> Option<(String, u32, f32)> {
        let info = self.hf_client.get_repo_info(repo_id).await.ok()?;
        if !info.siblings.iter().any(|f| f.rfilename == "config.json") {
            return None;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/config.json",
            repo_id, info.revision
        );
        let body = reqwest::Client::new().get(&url).send().await.ok()?;
        let config_json: serde_json::Value = body.json().await.ok()?;
        let architecture = extract_architecture(&config_json);
        let context_length = config_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as u32;
        let parameters_billion = estimate_parameters(&config_json);
        Some((architecture, context_length, parameters_billion))
    }
}

/// Map HuggingFace `architectures[0]` (e.g. `"LlamaForCausalLM"`) to the
/// short family name used by GGUF and llama.cpp (e.g. `"llama"`). Unknown
/// architectures fall back to `"unknown"` — model still works, but routing
/// hints based on family won't kick in.
fn extract_architecture(config: &serde_json::Value) -> String {
    let arch_raw = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    match arch_raw {
        "LlamaForCausalLM" => "llama",
        "MistralForCausalLM" => "mistral",
        "PhiForCausalLM" => "phi",
        "GemmaForCausalLM" => "gemma",
        "Qwen2ForCausalLM" => "qwen2",
        _ => "unknown",
    }
    .to_string()
}

/// Rough parameter count from transformer config geometry. The formula is the
/// standard "self-attention + MLP per layer + embedding matrix":
///   params ≈ L · (4·H² + 3·H·I) + V·H
/// where L=layers, H=hidden, I=intermediate, V=vocab. Returns billions of params.
///
/// Off by 5-15% vs actual (ignores biases, RoPE, lm_head sharing, MoE), but
/// accurate enough for VRAM estimation. Don't trust it for marketing copy.
pub fn estimate_parameters(config: &serde_json::Value) -> f32 {
    let h = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as f64;

    let l = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as f64;

    let v = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as f64;

    let default_intermediate = (4.0 * h) as u64;
    let i = config
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(default_intermediate) as f64;

    let params = l * (4.0 * h * h + 3.0 * h * i) + v * h;
    (params / 1e9) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn estimate_parameters_llama_7b_ballpark() {
        // Llama-2-7B canonical config — formula should yield ~7B.
        let cfg = json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        });
        let params = estimate_parameters(&cfg);
        // 6.5B ≤ estimate ≤ 7.5B (formula skips embeddings tying nuances)
        assert!((6.5..=7.5).contains(&params), "got {} B", params);
    }

    #[test]
    fn estimate_parameters_uses_default_intermediate() {
        // Without intermediate_size, formula uses 4*hidden_size.
        let cfg = json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        });
        let params = estimate_parameters(&cfg);
        // ~7B with the default 4x intermediate
        assert!(params > 5.0 && params < 10.0);
    }

    #[test]
    fn estimate_parameters_handles_empty_config() {
        // All defaults — should still produce a non-negative finite number.
        let cfg = json!({});
        let params = estimate_parameters(&cfg);
        assert!(params.is_finite());
        assert!(params >= 0.0);
    }

    #[test]
    fn extract_architecture_maps_known_arches() {
        assert_eq!(
            extract_architecture(&json!({"architectures": ["LlamaForCausalLM"]})),
            "llama"
        );
        assert_eq!(
            extract_architecture(&json!({"architectures": ["Qwen2ForCausalLM"]})),
            "qwen2"
        );
        assert_eq!(
            extract_architecture(&json!({"architectures": ["NewArchForCausalLM"]})),
            "unknown"
        );
        assert_eq!(extract_architecture(&json!({})), "unknown");
    }
}

