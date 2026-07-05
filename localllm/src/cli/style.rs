//! # Terminal styling — the single source of ANSI styling for the CLI.
//!
//! Every part of the CLI (the chat REPL, the markdown renderer, the setup
//! banner, the `list`/`ps`/`doctor` tables) draws its colors from here, so the
//! look stays consistent and there's one place to tune it. Colors degrade
//! gracefully on terminals that ignore ANSI.
//!
//! Also hosts the shared "thinking" spinner used wherever the CLI waits on a
//! slow operation (daemon startup, model cold-spawn), so the user always gets
//! live feedback instead of a silent hang.

use std::io::{self, Write};
use tokio::task::JoinHandle;
use tokio::time::Duration;

// Raw ANSI escape codes. Kept `pub` so call sites can compose them inline in
// `format!`/`print!` exactly as before, e.g. `format!("{BOLD}{CYAN}…{RESET}")`.
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const ITALIC: &str = "\x1b[3m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const MAGENTA: &str = "\x1b[35m";
pub const CYAN: &str = "\x1b[36m";

/// Wrap `s` green — used for success / healthy states ("OK", "ready").
pub fn ok(s: &str) -> String {
    format!("{GREEN}{s}{RESET}")
}

/// Wrap `s` yellow — used for warnings / non-fatal info.
pub fn warn(s: &str) -> String {
    format!("{YELLOW}{s}{RESET}")
}

/// Wrap `s` red — used for errors / missing requirements.
pub fn err(s: &str) -> String {
    format!("{RED}{s}{RESET}")
}

/// Wrap `s` dim — used for secondary text (headers, hints, separators).
pub fn dim(s: &str) -> String {
    format!("{DIM}{s}{RESET}")
}

/// Left-pad a (possibly already-colored) value to a fixed *visible* width.
///
/// `{:<N}` formatting counts raw bytes, so an ANSI-colored string throws column
/// alignment off (the escape codes are invisible but counted). This pads based
/// on `visible_len` — the number of printable columns the value occupies — then
/// appends the colored string, so tables line up regardless of color.
pub fn pad_colored(colored: &str, visible_len: usize, width: usize) -> String {
    let pad = width.saturating_sub(visible_len);
    format!("{colored}{}", " ".repeat(pad))
}

/// Frames for the braille "thinking" spinner. Shared so every spinner looks the
/// same across the CLI.
const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Start a dim braille spinner on the current line with `label` (e.g.
/// "thinking…", "starting localllm daemon…"). Returns a `JoinHandle`; abort it
/// and call [`clear_line`] once the awaited work finishes. The spinner rewrites
/// its own line with `\r`, so it never scrolls.
pub fn start_spinner(label: &'static str) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut i = 0usize;
        loop {
            print!("\r{DIM}{} {label}{RESET}", SPINNER_FRAMES[i % SPINNER_FRAMES.len()]);
            let _ = io::stdout().flush();
            i += 1;
            tokio::time::sleep(Duration::from_millis(80)).await;
        }
    })
}

/// Erase the current terminal line (carriage-return + clear-to-end-of-line).
/// Call after aborting a spinner so its last frame doesn't linger before the
/// real output prints.
pub fn clear_line() {
    print!("\r\x1b[K");
    let _ = io::stdout().flush();
}
