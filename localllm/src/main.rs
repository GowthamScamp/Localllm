//! # localllm — entry point
//!
//! Single binary serving two roles, dispatched by the first CLI arg:
//!   * `localllm serve`      — runs the long-lived HTTP daemon directly.
//!   * `localllm <anything>` — runs a CLI command that talks to a daemon over HTTP.
//!     If no daemon is reachable on the configured host:port, the CLI auto-spawns
//!     one as a detached background process and waits for `/health` to come up.
//!
//! The auto-spawn keeps `localllm` feeling like a single tool (no separate
//! `start-daemon` step) while keeping the daemon's lifecycle independent — once
//! launched, it survives parent shell exit, terminal close, and CLI process exit.

#![allow(dead_code)]

mod api;
mod cli;
mod compression;
mod config;
mod daemon;
mod downloader;
mod error;
mod gpu;
mod inference;
mod platform;
mod registry;
mod setup;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing_subscriber::EnvFilter;

use cli::commands::{Cli, Commands};
use config::Settings;

/// Tokio-driven entry point.
///
/// Flow:
///   1. Initialize tracing (defaults to `localllm=info`, overridable via `RUST_LOG`).
///   2. Parse CLI args and load settings (`~/.localllm/config.toml` + env overrides).
///   3. If the user typed `serve`, run the daemon in this process and never return
///      until shutdown.
///   4. Otherwise probe `/health` on the daemon URL. If it's down, fork a detached
///      copy of ourselves running `serve`, then poll `/health` for up to 5s.
///   5. Hand off to `cli::commands::execute` to perform the actual command.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Logging verbosity depends on the command. `serve` is the long-lived daemon
    // and wants info-level logs. Every other command is an interactive CLI call
    // where info/warn tracing (e.g. "Daemon not running…") would just be noise
    // that clutters otherwise-clean output — so default those to warn-only.
    // `RUST_LOG` always overrides, so power users can re-enable full logs.
    let default_filter = if matches!(cli.command, Commands::Serve) {
        "localllm=info"
    } else {
        "localllm=warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .init();

    let settings = Settings::load()?;

    // If this is the serve command, run daemon directly
    if matches!(cli.command, Commands::Serve) {
        return daemon::server::run(Arc::new(settings)).await;
    }

    // `completion`, `setup`, and `install` work without a running daemon — skip
    // auto-spawn so users don't pay the daemon startup cost (or trigger a build)
    // just to print a shell script, install the binary, or pre-build the engine.
    let needs_daemon = !matches!(
        cli.command,
        Commands::Completion { .. } | Commands::Setup { .. } | Commands::Install
    );

    // Resolve daemon URL
    let daemon_url = cli
        .daemon_url
        .clone()
        .unwrap_or_else(|| settings.daemon_url());

    // Check if daemon is running
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;

    let daemon_alive = client
        .get(format!("{}/health", daemon_url))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if needs_daemon && !daemon_alive {
        // First-run auto-setup, done in the FOREGROUND so the user sees the
        // (one-time, multi-minute) llama.cpp build progress instead of the
        // daemon silently hanging during startup. After this returns, the
        // daemon starts fast because llama.cpp is already built. No-op on
        // every subsequent run.
        if crate::setup::ensure_llama_cpp(&settings).await.is_err() {
            // Non-fatal: the daemon will retry and surface a clear error on the
            // endpoints that actually need llama.cpp. We don't block startup.
        }

        tracing::info!("Daemon not running, starting it in background...");

        // Show a spinner while the daemon comes up so the user gets feedback
        // instead of a silent pause. Only on a TTY (keeps piped output clean).
        use std::io::IsTerminal;
        let spinner = if std::io::stdout().is_terminal() {
            Some(cli::style::start_spinner("starting localllm…"))
        } else {
            None
        };

        let exe = std::env::current_exe()?;

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // DETACHED_PROCESS (0x08) | CREATE_NEW_PROCESS_GROUP (0x200) — daemon
            // outlives this CLI process and is not killed by parent console close.
            const DETACHED_PROCESS: u32 = 0x00000008;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
            std::process::Command::new(&exe)
                .arg("serve")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
                .spawn()?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // setsid() puts the child in a new session, detaching it from this
            // CLI's controlling terminal so SIGHUP doesn't take it down.
            extern "C" {
                fn setsid() -> i32;
            }
            unsafe {
                std::process::Command::new(&exe)
                    .arg("serve")
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .pre_exec(|| {
                        setsid();
                        Ok(())
                    })
                    .spawn()?;
            }
        }

        // Poll the health endpoint with a tight loop instead of a fixed 2s sleep —
        // small daemons become reachable in ~200ms, not 2s.
        let mut waited_ms = 0u64;
        let mut alive = false;
        while waited_ms < 5000 {
            sleep(Duration::from_millis(100)).await;
            waited_ms += 100;
            if client
                .get(format!("{}/health", daemon_url))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                alive = true;
                break;
            }
        }
        if let Some(s) = spinner {
            s.abort();
            cli::style::clear_line();
        }
        if !alive {
            tracing::warn!("Daemon may not have started yet, proceeding anyway...");
        }
    }

    cli::commands::execute(cli, Arc::new(settings), daemon_url).await
}
