//! # HuggingFace API client
//!
//! Talks to `huggingface.co` for two things:
//!   1. **Metadata** — `GET /api/models/{repo_id}` returns the file listing
//!      and the commit hash (X-Repo-Commit header).
//!   2. **File download** — `GET /{repo_id}/resolve/{rev}/{filename}` streams
//!      the file bytes. Supports HTTP range resume for interrupted transfers.
//!
//! Two reqwest clients are kept because metadata calls want a short timeout
//! (fail fast on stuck APIs) while file downloads must run without an overall
//! timeout (multi-GB files take minutes).

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use reqwest::header::{AUTHORIZATION, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoFile {
    pub rfilename: String,
    pub size: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct RepoInfo {
    pub id: String,
    pub revision: String,
    pub siblings: Vec<RepoFile>,
}

pub struct HuggingFaceClient {
    /// Used for metadata API calls (short connect + read timeout).
    api_client: reqwest::Client,
    /// Used for file downloads — no overall timeout, only a connect timeout,
    /// because large files can legitimately take minutes to stream.
    download_client: reqwest::Client,
    token: Option<String>,
}

impl HuggingFaceClient {
    /// Build a new client. The optional `token` is attached to every request
    /// as `Authorization: Bearer ...` — required for gated/private models.
    pub fn new(token: Option<String>) -> Result<Self> {
        // Normalize: an empty / whitespace-only token is treated as no token.
        // Sending `Authorization: Bearer ` would make HF return 401 even for
        // public models.
        let token = token
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty());

        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref t) = token {
            headers.insert(
                AUTHORIZATION,
                format!("Bearer {}", t)
                    .parse()
                    .map_err(|e| anyhow!("Invalid HF token format: {}", e))?,
            );
        }

        let api_client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers.clone())
            .build()?;

        // Download client: only connect timeout, no total timeout.
        // reqwest's .timeout() caps the *entire* request including streaming body,
        // which would kill multi-GB downloads on slow connections.
        let download_client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .default_headers(headers)
            .build()?;

        Ok(Self {
            api_client,
            download_client,
            token,
        })
    }

    /// Fetch repo metadata: commit hash + sibling file list. The commit hash
    /// comes from the `X-Repo-Commit` response header; the sibling list comes
    /// from the JSON body's `siblings[].rfilename` field. We tolerate missing
    /// `size` since some HF endpoints omit it for LFS pointers.
    pub async fn get_repo_info(&self, repo_id: &str) -> Result<RepoInfo> {
        // `?blobs=true` makes HF include each file's real byte size in the
        // siblings list (the default response omits it). Without sizes the
        // cached-skip fast path can't fire and a fully-downloaded file gets a
        // bogus resume Range → HTTP 416. Sizes also drive the progress bar.
        let url = format!("https://huggingface.co/api/models/{}?blobs=true", repo_id);
        tracing::debug!("Fetching repo info: {}", url);

        let resp = self.api_client.get(&url).send().await?;

        let revision = resp
            .headers()
            .get("X-Repo-Commit")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("main")
            .to_string();

        if !resp.status().is_success() {
            return Err(anyhow!(
                "HuggingFace API error {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            ));
        }

        let body: serde_json::Value = resp.json().await?;

        let siblings: Vec<RepoFile> = body
            .get("siblings")
            .and_then(|s| s.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let rfilename = item
                            .get("rfilename")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if rfilename.is_empty() {
                            return None;
                        }
                        // Plain files report `size`; LFS files (large .gguf /
                        // .safetensors) report it under `lfs.size`. Fall back to
                        // the latter so large weights still get an accurate size.
                        let size = item
                            .get("size")
                            .and_then(|v| v.as_u64())
                            .or_else(|| {
                                item.get("lfs")
                                    .and_then(|l| l.get("size"))
                                    .and_then(|v| v.as_u64())
                            });
                        Some(RepoFile { rfilename, size })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(RepoInfo {
            id: repo_id.to_string(),
            revision,
            siblings,
        })
    }

    /// Return only the files we actually want to download for a given repo.
    /// Filters by an allow-list (config + tokenizer + safetensors) and a
    /// deny-list (legacy `.pt`/`.h5`/etc. duplicates). Also enforces
    /// `is_safe_filename` against path-traversal attacks from a malicious
    /// or compromised HF mirror.
    pub async fn list_files(&self, repo_id: &str, _revision: &str) -> Result<Vec<RepoFile>> {
        let info = self.get_repo_info(repo_id).await?;

        // Static allow/deny lists — single iter pass per file, no per-call allocation.
        const ALLOWED_EXACT: &[&str] = &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ];
        const ALLOWED_SUFFIXES: &[&str] = &[".safetensors"];
        const DENIED_SUFFIXES: &[&str] = &[".msgpack", ".h5", ".ot", ".npz", ".pt"];

        let mut out = Vec::with_capacity(info.siblings.len().min(32));
        for f in info.siblings.into_iter() {
            let n = f.rfilename.as_str();
            if DENIED_SUFFIXES.iter().any(|s| n.ends_with(s)) {
                continue;
            }
            if !crate::registry::manifest::is_safe_filename(n) {
                tracing::warn!("Skipping unsafe filename from HF: {}", n);
                continue;
            }
            let allowed = ALLOWED_EXACT.contains(&n)
                || ALLOWED_SUFFIXES.iter().any(|s| n.ends_with(s));
            if allowed {
                out.push(f);
            }
        }

        Ok(out)
    }

    /// List the `.gguf` files available in a repo, with their sizes. Used by the
    /// GGUF auto-download path so we can pick the smallest file matching the
    /// requested quant level without pulling torch / running a conversion.
    ///
    /// Returns an empty Vec if the repo has no GGUF files (e.g. it only ships
    /// safetensors) — callers treat that as "fall back to other strategies".
    pub async fn list_gguf_files(&self, repo_id: &str) -> Result<Vec<RepoFile>> {
        let info = self.get_repo_info(repo_id).await?;
        let mut out: Vec<RepoFile> = info
            .siblings
            .into_iter()
            .filter(|f| {
                let n = f.rfilename.to_lowercase();
                n.ends_with(".gguf")
                    && crate::registry::manifest::is_safe_filename(&f.rfilename)
                    // Skip split-file shards (e.g. "-00001-of-00002.gguf"); we'd
                    // need to download & merge all parts. Single-file quants only.
                    && !n.contains("-of-")
            })
            .collect();
        // Stable order: smaller files first so the quant picker is deterministic.
        out.sort_by_key(|f| f.size.unwrap_or(u64::MAX));
        Ok(out)
    }

    /// Probe whether a repo exists and is reachable (HTTP 200 on its metadata).
    /// Used to test candidate GGUF mirror repos before committing to a download.
    pub async fn repo_exists(&self, repo_id: &str) -> bool {
        let url = format!("https://huggingface.co/api/models/{}", repo_id);
        matches!(self.api_client.get(&url).send().await, Ok(r) if r.status().is_success())
    }

    /// Download one file with three robustness layers:
    ///   1. **Cached fast path** — if `dest_path` exists at `expected_size`,
    ///      skip the request entirely.
    ///   2. **Retry with exponential backoff** — 3 attempts (2s, 4s, 8s).
    ///      Each retry resets the progress bar and deletes the partial file.
    ///   3. **Resume on retry** — `attempt_download` honors any existing
    ///      partial bytes via HTTP Range, so a network blip doesn't restart
    ///      a 10 GB file.
    // Each parameter is a distinct, necessary input to the download (no natural
    // struct to group them); the 8th over clippy's default-7 limit is benign.
    #[allow(clippy::too_many_arguments)]
    pub async fn download_file(
        &self,
        repo_id: &str,
        revision: &str,
        filename: &str,
        dest_path: &Path,
        expected_size: Option<u64>,
        expected_sha256: Option<&str>,
        progress_bar: &ProgressBar,
    ) -> Result<()> {
        // Cached fast path: if the file already exists with the exact expected
        // size, skip download entirely. This is a major win for repeated `pull`
        // commands and for restoring after a process restart.
        if let Some(expected) = expected_size {
            if dest_path.exists() {
                if let Ok(meta) = tokio::fs::metadata(dest_path).await {
                    if meta.len() == expected {
                        tracing::debug!(
                            "{} already present at expected size ({} bytes), skipping",
                            filename,
                            expected
                        );
                        progress_bar.set_position(expected);
                        progress_bar.finish_with_message(format!("cached: {}", filename));
                        return Ok(());
                    }
                }
            }
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo_id, revision, filename
        );

        let mut last_err = None;

        for attempt in 0u32..3 {
            if attempt > 0 {
                let delay = 2u64.pow(attempt);
                tracing::info!(
                    "Retrying download of {} in {}s (attempt {})",
                    filename,
                    delay,
                    attempt + 1
                );
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                progress_bar.set_position(0);
            }

            match self
                .attempt_download(&url, dest_path, expected_sha256, progress_bar)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(
                        "Download attempt {} failed for {}: {}",
                        attempt + 1,
                        filename,
                        e
                    );
                    last_err = Some(e);
                    let _ = tokio::fs::remove_file(dest_path).await;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("Download failed after 3 attempts: {}", filename)))
    }

    /// One actual download attempt. Issues a `GET` with optional `Range:` for
    /// resume, streams the body to `dest_path`, computes SHA-256 on fresh
    /// downloads (skipped for resumes since we can't hash bytes we didn't see).
    async fn attempt_download(
        &self,
        url: &str,
        dest_path: &Path,
        expected_sha256: Option<&str>,
        progress_bar: &ProgressBar,
    ) -> Result<()> {
        let existing_size = if dest_path.exists() {
            tokio::fs::metadata(dest_path).await?.len()
        } else {
            0
        };

        let is_resume = existing_size > 0;
        let mut request = self.download_client.get(url);

        if let Some(ref t) = self.token {
            request = request.header(AUTHORIZATION, format!("Bearer {}", t));
        }

        if is_resume {
            tracing::debug!("Resuming download from byte {}", existing_size);
            request = request.header(RANGE, format!("bytes={}-", existing_size));
        }

        let resp = request.send().await?;
        let status = resp.status();

        // HTTP 416: resume offset is at/past EOF → the local file is already
        // complete. Verify the hash if we have one, otherwise accept it.
        if status.as_u16() == 416 {
            tracing::debug!(
                "Range not satisfiable for {} — local file already complete",
                url
            );
            progress_bar.set_position(existing_size);
            return Ok(());
        }

        if !status.is_success() && status.as_u16() != 206 {
            return Err(anyhow!("HTTP {} for {}", status, url));
        }

        // If we asked for a range but the server replied 200 (full body), it
        // ignored the Range header and is streaming the whole file from byte 0.
        // Appending to the existing partial would corrupt the file — treat this
        // as a fresh download instead (truncate + write from the start).
        let server_honored_range = status.as_u16() == 206;
        let is_resume = is_resume && server_honored_range;

        if let Some(parent) = dest_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = if is_resume {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(dest_path)
                .await?
        } else {
            tokio::fs::File::create(dest_path).await?
        };

        // Only hash on a fresh download — a resumed file can't be verified incrementally.
        let mut hasher = if !is_resume && expected_sha256.is_some() {
            Some(Sha256::new())
        } else {
            None
        };

        let mut stream = resp.bytes_stream();
        use futures::StreamExt;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            file.write_all(&chunk).await?;
            if let Some(ref mut h) = hasher {
                h.update(&chunk);
            }
            progress_bar.inc(chunk.len() as u64);
            tracing::debug!("Downloaded chunk {} bytes from {}", chunk.len(), url);
        }

        file.flush().await?;

        if let (Some(hasher), Some(expected)) = (hasher, expected_sha256) {
            let computed = hex::encode(hasher.finalize());
            if computed != expected {
                return Err(anyhow!(
                    "SHA256 mismatch for {}: expected {}, got {}",
                    url,
                    expected,
                    computed
                ));
            }
        }

        Ok(())
    }

    /// Download a single file, reporting progress via a callback instead of an
    /// `indicatif` bar. `on_progress(downloaded, total)` is invoked roughly once
    /// per network chunk — the daemon uses this to stream byte-progress frames to
    /// the CLI so the user sees a live bar. `total` is the expected size (0 if
    /// unknown). Supports the same cached-skip fast path as `download_file`.
    pub async fn download_file_with_callback<F>(
        &self,
        repo_id: &str,
        revision: &str,
        filename: &str,
        dest_path: &Path,
        expected_size: Option<u64>,
        mut on_progress: F,
    ) -> Result<()>
    where
        F: FnMut(u64, u64),
    {
        let total = expected_size.unwrap_or(0);

        // Cached fast path — already fully downloaded.
        if let Some(expected) = expected_size {
            if dest_path.exists() {
                if let Ok(meta) = tokio::fs::metadata(dest_path).await {
                    if meta.len() == expected {
                        on_progress(expected, total);
                        return Ok(());
                    }
                }
            }
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo_id, revision, filename
        );

        let existing_size = if dest_path.exists() {
            tokio::fs::metadata(dest_path).await?.len()
        } else {
            0
        };
        let is_resume = existing_size > 0 && expected_size.map(|e| existing_size < e).unwrap_or(true);

        let mut request = self.download_client.get(&url);
        if let Some(ref t) = self.token {
            request = request.header(AUTHORIZATION, format!("Bearer {}", t));
        }
        if is_resume {
            request = request.header(RANGE, format!("bytes={}-", existing_size));
        }

        let resp = request.send().await?;
        let status = resp.status();

        // HTTP 416 Range Not Satisfiable: we asked to resume from `existing_size`
        // but the server says that offset is at/beyond the end — i.e. the local
        // file is already complete. This happens when the HF metadata didn't give
        // us a size (so the cached fast-path above couldn't fire) but the file is
        // in fact fully downloaded. Treat it as success.
        if status.as_u16() == 416 {
            tracing::debug!(
                "Range not satisfiable for {} — local file already complete ({} bytes)",
                url,
                existing_size
            );
            on_progress(existing_size, total.max(existing_size));
            return Ok(());
        }

        if !status.is_success() && status.as_u16() != 206 {
            return Err(anyhow!("HTTP {} for {}", status, url));
        }

        // Server ignored our Range header (replied 200 not 206) → it's sending
        // the full file. Overwrite from byte 0 rather than appending to the
        // partial, which would corrupt the download.
        let server_honored_range = status.as_u16() == 206;
        let is_resume = is_resume && server_honored_range;

        if let Some(parent) = dest_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = if is_resume {
            tokio::fs::OpenOptions::new().append(true).open(dest_path).await?
        } else {
            tokio::fs::File::create(dest_path).await?
        };

        let mut downloaded = if is_resume { existing_size } else { 0 };
        on_progress(downloaded, total);

        let mut stream = resp.bytes_stream();
        use futures::StreamExt;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            on_progress(downloaded, total);
        }
        file.flush().await?;
        Ok(())
    }
}
