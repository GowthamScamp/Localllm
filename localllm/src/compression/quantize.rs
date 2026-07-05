//! # Quantization pipeline
//!
//! Two-stage conversion from HuggingFace safetensors to a quantized GGUF:
//!   1. `python convert_hf_to_gguf.py --outtype f16` — produces an F16 GGUF.
//!   2. `llama-quantize <in.gguf> <out.gguf> <level>` — quantizes to target.
//!
//! Both stages shell out to llama.cpp's tooling, so `settings.llama_cpp_dir`
//! must point at a built llama.cpp checkout. The intermediate F16 file is
//! deleted on success — it's typically 2x the size of the final quant.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::config::Settings;
use crate::registry::{ModelManifest, QuantizationLevel};

/// Wraps the llama.cpp conversion + quantization scripts. Stateless apart
/// from the directory pointer; one instance shared across the daemon.
pub struct CompressionEngine {
    llama_cpp_dir: PathBuf,
    settings: Arc<Settings>,
}

impl CompressionEngine {
    /// Construct from the llama.cpp source dir and process-wide settings.
    /// Does NOT validate the directory exists yet — that happens lazily on
    /// the first `quantize()` call so the daemon can start even with a stale
    /// or pending `llama_cpp_dir`.
    pub fn new(llama_cpp_dir: PathBuf, settings: Arc<Settings>) -> Self {
        Self {
            llama_cpp_dir,
            settings,
        }
    }

    /// Quantize a model end-to-end and return the path of the final GGUF.
    /// Long-running (minutes for 7B+); awaitable so the HTTP handler can
    /// stream progress in the future. Errors include the convert script's
    /// or quantize binary's full stderr to aid debugging.
    pub async fn quantize(
        &self,
        manifest: &ModelManifest,
        level: QuantizationLevel,
    ) -> Result<PathBuf> {
        let alias = &manifest.alias;
        let level_str = level.to_llama_str();

        // Step 1 — Convert to GGUF F16
        let convert_script = self.llama_cpp_dir.join("convert_hf_to_gguf.py");
        if !convert_script.exists() {
            return Err(anyhow!(
                "convert_hf_to_gguf.py not found at {:?}",
                convert_script
            ));
        }

        let f16_path = self.settings.gguf_dir.join(format!("{}-F16.gguf", alias));
        tracing::info!("Converting {} to GGUF F16 at {:?}", alias, f16_path);

        // Resolve the Python interpreter for this OS (python3 / python / py).
        let python = crate::platform::python_command();

        spawn_and_log(
            Command::new(&python)
                .arg(&convert_script)
                .arg(&manifest.local_path)
                .arg("--outtype")
                .arg("f16")
                .arg("--outfile")
                .arg(&f16_path),
        )
        .await
        .map_err(|e| anyhow!("convert_hf_to_gguf.py failed for {}: {}", alias, e))?;

        // Step 2 — Quantize to target level. Cross-platform binary resolution.
        let quantize_bin =
            crate::platform::resolve_llama_binary(&self.llama_cpp_dir, "llama-quantize")
                .ok_or_else(|| {
                    anyhow!(
                        "llama-quantize not found under {:?}/build/bin/",
                        self.llama_cpp_dir
                    )
                })?;

        let output_path = self
            .settings
            .gguf_dir
            .join(format!("{}-{}.gguf", alias, level_str));

        tracing::info!("Quantizing {} → {} at {:?}", alias, level_str, output_path);

        spawn_and_log(
            Command::new(&quantize_bin)
                .arg(&f16_path)
                .arg(&output_path)
                .arg(level_str),
        )
        .await
        .map_err(|e| anyhow!("llama-quantize failed for {}: {}", alias, e))?;

        // Step 3 — Remove intermediate F16 file
        tokio::fs::remove_file(&f16_path)
            .await
            .map_err(|e| anyhow!("Failed to remove intermediate F16 {:?}: {}", f16_path, e))?;

        tracing::info!("Quantization complete: {:?}", output_path);
        Ok(output_path)
    }
}

/// Spawn `cmd`, stream stdout to `tracing::debug`, capture stderr via an mpsc
/// channel, await exit. On non-zero exit, returns `Err` with the full captured
/// stderr text so handlers can surface it back to the user.
///
/// Why mpsc for stderr: spawning a task that calls `lines.next_line().await`
/// avoids blocking the main thread on the child's stderr pipe, while still
/// letting us collect all lines after the child exits. A bounded channel would
/// risk deadlock if the child writes too much before we drain.
pub async fn spawn_and_log(cmd: &mut Command) -> Result<()> {
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn process: {}", e))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Failed to capture stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Failed to capture stderr"))?;

    // Capture stderr for error reporting via mpsc channel — no per-line lock
    // contention. The reader task sends each line; we drain into a Vec only after
    // the process exits and the channel is closed.
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let stdout_task = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("[stdout] {}", line);
        }
    });

    let stderr_task = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("[stderr] {}", line);
            // Best-effort send; if the receiver has been dropped we just log.
            let _ = tx.send(line);
        }
        // Drop tx → close the channel when stderr reader exits
    });

    let status = child
        .wait()
        .await
        .map_err(|e| anyhow!("Failed to wait for child process: {}", e))?;

    // Wait for stderr reader to finish (so the channel is closed) before draining.
    let _ = stdout_task.await;
    let _ = stderr_task.await;

    if !status.success() {
        let mut stderr_lines: Vec<String> = Vec::new();
        while let Ok(line) = rx.try_recv() {
            stderr_lines.push(line);
        }
        return Err(anyhow!(
            "Process exited with code {:?}\nStderr:\n{}",
            status.code(),
            stderr_lines.join("\n")
        ));
    }

    Ok(())
}
