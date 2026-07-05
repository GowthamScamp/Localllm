//! # First-run auto-setup
//!
//! Makes `localllm` work after nothing more than `cargo build`. The first time
//! the daemon needs llama.cpp and can't find a usable build, it clones and
//! builds llama.cpp into `~/.localllm/llama.cpp` automatically, then points the
//! running config at it. No manual `git clone`, no `cmake`, no editing
//! `config.toml`.
//!
//! Everything here is cross-platform (Windows + Linux + macOS):
//!   * tool detection probes `git`/`cmake` with `--version`;
//!   * the build invokes CMake the same way on every OS (the `--target` form
//!     builds only `llama-server` + `llama-quantize`, which is all we need and
//!     keeps the build minutes-not-tens-of-minutes);
//!   * the resulting binary is resolved through `platform::resolve_llama_binary`
//!     so the `.exe` suffix is handled transparently.
//!
//! Idempotent: if a usable `llama-server` already exists under the managed dir
//! (or a user-configured `llama_cpp_dir`), setup is a no-op.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::Settings;

/// Default upstream for llama.cpp. Pinned to the canonical repo; users who want
/// a fork can pre-build it and set `llama_cpp_dir` to skip auto-setup entirely.
const LLAMA_CPP_REPO: &str = "https://github.com/ggerganov/llama.cpp";

/// Where auto-setup clones + builds llama.cpp: `~/.localllm/llama.cpp`.
/// Lives under the data dir so it follows the user's home and is easy to find.
pub fn managed_llama_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".localllm")
        .join("llama.cpp")
}

/// Does this directory contain a usable, built llama.cpp (i.e. a runnable
/// `llama-server`)? This is the single source of truth for "is setup done".
pub fn is_llama_built(dir: &Path) -> bool {
    crate::platform::resolve_llama_binary(dir, "llama-server").is_some()
}

/// Resolve a usable llama.cpp directory, building one on first run if needed.
///
/// Resolution order:
///   1. A user-configured `llama_cpp_dir` that's actually built → use it.
///   2. The managed `~/.localllm/llama.cpp` if already built → use it.
///   3. Otherwise clone + build into the managed dir, then use it.
///
/// Returns the directory to set as `llama_cpp_dir`. Errors only when the build
/// genuinely can't proceed (missing git/cmake/compiler) — and those errors
/// carry copy-pasteable install hints.
pub async fn ensure_llama_cpp(settings: &Settings) -> Result<PathBuf> {
    ensure_llama_cpp_inner(settings, false).await
}

/// Like [`ensure_llama_cpp`] but `force_rebuild = true` ignores any existing
/// build and rebuilds from scratch — used by `localllm setup --rebuild` so a
/// user who installs a GPU toolkit after the first (CPU-only) build can upgrade.
pub async fn ensure_llama_cpp_rebuild(settings: &Settings) -> Result<PathBuf> {
    ensure_llama_cpp_inner(settings, true).await
}

async fn ensure_llama_cpp_inner(settings: &Settings, force_rebuild: bool) -> Result<PathBuf> {
    use crate::cli::style::{BOLD, CYAN, DIM, GREEN, RESET};

    if !force_rebuild {
        // 1. Respect an existing, working user config.
        if let Some(dir) = settings.llama_cpp_dir.as_ref() {
            if is_llama_built(dir) {
                return Ok(dir.clone());
            }
        }
        // 2. Managed dir already built?
        let managed = managed_llama_dir();
        if is_llama_built(&managed) {
            return Ok(managed);
        }
    }

    // 3. Build it. Detect the best available GPU backend up-front so the banner
    //    can name it, then run the blocking git/cmake work off the async runtime.
    let managed = managed_llama_dir();
    let (_flags, backend) = gpu_cmake_flags();
    eprintln!("\n{CYAN}{BOLD}── localllm first-run setup ──{RESET}");
    eprintln!(
        "{DIM}Building the inference engine (llama.cpp) with {backend} acceleration. \
         One time only, ~3-8 min.{RESET}\n"
    );
    let managed_for_build = managed.clone();
    tokio::task::spawn_blocking(move || build_llama_cpp(&managed_for_build))
        .await
        .map_err(|e| anyhow!("setup task panicked: {}", e))??;
    eprintln!("\n{GREEN}{BOLD}✓ Setup complete.{RESET}\n");

    if !is_llama_built(&managed) {
        return Err(anyhow!(
            "llama.cpp build finished but llama-server is still missing under {:?}",
            managed
        ));
    }
    Ok(managed)
}

/// Clone (if needed) and build llama.cpp into `dir`. Synchronous; intended to
/// run inside `spawn_blocking`. Streams the cmake build output to the daemon log
/// at info level so progress is observable via `localllm logs`/`tail`.
///
/// Picks the best GPU backend available on the machine (CUDA, else Vulkan, else
/// CPU). If a GPU-accelerated build fails (half-installed SDK, arch mismatch),
/// it falls back to a CPU build so first-run setup can never be bricked by a
/// broken GPU toolchain.
fn build_llama_cpp(dir: &Path) -> Result<()> {
    preflight_tools()?;

    // Clone if the source isn't there yet. A partial/old checkout is reused;
    // we only clone when the directory doesn't exist at all.
    if !dir.join("CMakeLists.txt").exists() {
        if let Some(parent) = dir.parent() {
            std::fs::create_dir_all(parent)?;
        }
        tracing::info!("Cloning llama.cpp into {:?} (one-time, ~1-2 min)", dir);
        run(
            Command::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg(LLAMA_CPP_REPO)
                .arg(dir),
            "git clone",
        )?;
    }

    let (gpu_flags, backend) = gpu_cmake_flags();

    // First attempt: with the detected GPU backend (if any).
    match configure_and_build(dir, &gpu_flags, backend) {
        Ok(()) => {
            tracing::info!("llama.cpp setup complete at {:?} ({} backend)", dir, backend);
            Ok(())
        }
        Err(e) if !gpu_flags.is_empty() => {
            // GPU build failed — fall back to CPU so setup still succeeds. cmake
            // reconfiguring the same build/ dir with different flags is idempotent.
            tracing::warn!(
                "{} build failed ({}); falling back to a CPU build",
                backend,
                e
            );
            configure_and_build(dir, &[], "CPU")?;
            tracing::info!("llama.cpp setup complete at {:?} (CPU fallback)", dir);
            Ok(())
        }
        Err(e) => Err(e), // CPU build failed outright — nothing to fall back to.
    }
}

/// Run `cmake` configure + build for `dir`, passing `extra_flags` (e.g. GPU
/// backend selectors). Builds only `llama-server` + `llama-quantize`.
fn configure_and_build(dir: &Path, extra_flags: &[String], backend: &str) -> Result<()> {
    tracing::info!("Configuring llama.cpp build (cmake, {} backend)...", backend);
    let mut configure = Command::new("cmake");
    configure
        .arg("-B")
        .arg(dir.join("build"))
        .arg("-S")
        .arg(dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DLLAMA_CURL=OFF");
    for flag in extra_flags {
        configure.arg(flag);
    }
    run(&mut configure, "cmake configure")?;

    // Build only what we use: llama-server + llama-quantize. Far faster than a
    // full build of every example/tool. `-j` parallelism uses all cores.
    tracing::info!("Building llama-server + llama-quantize (the slow part, ~3-8 min)...");
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .to_string();
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(dir.join("build"))
            .arg("--config")
            .arg("Release")
            .arg("-j")
            .arg(&jobs)
            .arg("--target")
            .arg("llama-server")
            .arg("--target")
            .arg("llama-quantize"),
        "cmake build",
    )?;
    Ok(())
}

/// Detect the best GPU backend whose *build toolkit* is installed, returning the
/// cmake flags to enable it plus a human label. Ubuntu + Windows only:
///   1. CUDA  — if `nvcc` is on PATH (NVIDIA CUDA Toolkit installed).
///   2. Vulkan — else if `glslc` is on PATH (the shader compiler llama.cpp's
///      Vulkan backend needs); a portable fallback covering AMD/Intel/NVIDIA.
///   3. CPU   — otherwise.
///
/// Note: this probes the *build* toolchain, not the runtime GPU. A box can have
/// `nvidia-smi` (driver) but no `nvcc` (toolkit) — only the latter can compile a
/// CUDA build, so we key off it.
fn gpu_cmake_flags() -> (Vec<String>, &'static str) {
    if tool_present("nvcc") {
        (vec!["-DGGML_CUDA=ON".to_string()], "CUDA")
    } else if tool_present("glslc") {
        (vec!["-DGGML_VULKAN=ON".to_string()], "Vulkan")
    } else {
        (Vec::new(), "CPU")
    }
}

/// Verify `git` and `cmake` are available before attempting a build, so we fail
/// with an actionable message instead of a cryptic spawn error halfway through.
fn preflight_tools() -> Result<()> {
    let mut missing = Vec::new();
    if !tool_present("git") {
        missing.push("git");
    }
    if !tool_present("cmake") {
        missing.push("cmake");
    }
    if missing.is_empty() {
        return Ok(());
    }

    let hint = if cfg!(windows) {
        "Install with: winget install Git.Git Kitware.CMake  (and a C++ compiler, e.g. Visual Studio Build Tools)"
    } else if cfg!(target_os = "macos") {
        "Install with: brew install git cmake"
    } else {
        "Install with: sudo apt-get install -y git cmake build-essential"
    };
    Err(anyhow!(
        "Automatic llama.cpp setup needs {} but they're not on PATH.\n{}",
        missing.join(" and "),
        hint
    ))
}

/// True if `tool --version` runs successfully (tool is installed + on PATH).
fn tool_present(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a build command, inheriting stdio so output lands in the daemon log.
/// Returns a clean error (with the step name) on non-zero exit.
fn run(cmd: &mut Command, step: &str) -> Result<()> {
    let status = cmd
        .status()
        .map_err(|e| anyhow!("failed to start {} ({}). Is it installed?", step, e))?;
    if !status.success() {
        return Err(anyhow!(
            "{} failed (exit {:?}). See the daemon log for details.",
            step,
            status.code()
        ));
    }
    Ok(())
}
