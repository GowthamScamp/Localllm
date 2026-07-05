//! # Cross-platform helpers
//!
//! Centralizes every OS-dependent decision so the rest of the codebase stays
//! platform-agnostic. Covers:
//!   * executable-name suffixes (`.exe` on Windows, none on Unix);
//!   * resolving a binary inside llama.cpp's `build/bin/` regardless of suffix;
//!   * picking the right Python interpreter (`python` vs `python3` vs `py`).
//!
//! Keeping this in one module means adding macOS/other targets later is a
//! single-file change, and the inference/quantize code never sprinkles
//! `cfg!(windows)` around.

use std::path::{Path, PathBuf};

/// Append the platform executable suffix to a bare binary name.
/// `"llama-server"` → `"llama-server.exe"` on Windows, unchanged on Unix.
pub fn exe_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

/// Resolve a llama.cpp build binary (e.g. `llama-server`, `llama-quantize`)
/// inside `<llama_cpp_dir>/build/bin/`. Tries the platform-suffixed name first,
/// then the bare name as a fallback (some builds/installs drop the `.exe`, and
/// cross-compiled Unix binaries never have it). Returns the first path that
/// exists, or `None` if neither is present.
pub fn resolve_llama_binary(llama_cpp_dir: &Path, stem: &str) -> Option<PathBuf> {
    let bin_dir = llama_cpp_dir.join("build").join("bin");
    let candidates = [exe_name(stem), stem.to_string()];
    for name in candidates {
        let p = bin_dir.join(&name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Pick a working Python interpreter command.
///
/// Order of preference:
///   * Unix: `python3`, then `python`.
///   * Windows: `python`, then `py` (the launcher), then `python3`.
///
/// We probe each by running `<cmd> --version` and taking the first that
/// succeeds. Falls back to a sensible default (`python3` on Unix, `python` on
/// Windows) if none respond — so callers still get a command to try, and the
/// spawn error surfaces normally instead of us guessing wrong silently.
pub fn python_command() -> String {
    let candidates: &[&str] = if cfg!(windows) {
        &["python", "py", "python3"]
    } else {
        &["python3", "python"]
    };

    for cmd in candidates {
        if probe_command(cmd, &["--version"]) {
            return cmd.to_string();
        }
    }

    // Nothing responded; return the conventional default for the OS.
    if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

/// Return true if `cmd args...` runs and exits successfully. Used to probe for
/// tool availability without caring about output.
fn probe_command(cmd: &str, args: &[&str]) -> bool {
    std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exe_name_adds_suffix_on_windows_only() {
        let n = exe_name("llama-server");
        if cfg!(windows) {
            assert_eq!(n, "llama-server.exe");
        } else {
            assert_eq!(n, "llama-server");
        }
    }

    #[test]
    fn resolve_returns_none_for_missing_dir() {
        let dir = PathBuf::from("/definitely/not/a/real/llama/dir");
        assert!(resolve_llama_binary(&dir, "llama-server").is_none());
    }

    #[test]
    fn python_command_returns_nonempty() {
        // Always returns a command string even if no Python is installed.
        assert!(!python_command().is_empty());
    }
}
