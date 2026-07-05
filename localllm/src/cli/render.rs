//! # Terminal markdown + LaTeX renderer
//!
//! Models answer in Markdown with LaTeX math. Dumped raw to a terminal that's
//! ugly: literal `**bold**`, `### Heading`, `$x^2$`, ` ```code``` `. This module
//! turns that into readable ANSI-styled output and converts common LaTeX math
//! into Unicode (superscripts, subscripts, Greek letters, common symbols).
//!
//! ## Streaming model
//!
//! Generation streams token-by-token, but Markdown needs whole lines to render
//! (you can't know `*` opens bold until its partner arrives). So [`MarkdownStream`]
//! buffers text and flushes **one complete line at a time** as newlines arrive,
//! rendering each line fully. The final partial line is flushed on [`finish`].
//! This keeps output responsive (line latency, not whole-message latency) while
//! still rendering correctly.
//!
//! Fenced code blocks (```) are detected and passed through verbatim (dimmed,
//! no inline-markdown processing) so code isn't mangled.
//!
//! No external crates — small, dependency-free, and predictable.

use std::io::{self, Write};

// ANSI styles come from the shared style module so the whole CLI matches.
use crate::cli::style::{BOLD, CYAN, DIM, GREEN, ITALIC, MAGENTA, RESET, YELLOW};

/// Line-buffered Markdown+LaTeX renderer for a streaming response. Feed it raw
/// model text with [`push`]; it prints fully-rendered lines as they complete.
#[derive(Default)]
pub struct MarkdownStream {
    /// Buffers the current line until we know its block type (heading? list?).
    /// Once the block prefix is resolved, prose streams through live and only
    /// the trailing `pending_span` is held back.
    line_buf: String,
    /// True while inside a ``` fenced code block (verbatim passthrough).
    in_code_block: bool,
    /// True while inside a multi-line display-math block opened by `\[` or `$$`
    /// on its own line and closed by `\]` / `$$`. Lines inside are accumulated
    /// and rendered as one LaTeX expression, since `\frac{a}{b}` etc. commonly
    /// span the block.
    in_math_block: bool,
    /// Accumulated display-math content while `in_math_block`.
    math_buf: String,
    /// Whether the current line's block-level prefix has been emitted yet. Until
    /// it is, we buffer so we can detect `# `, `- `, `1. ` etc. at line start.
    line_started: bool,
    /// Bytes of the current line already streamed to the terminal — so when we
    /// decide a line was actually a heading/bullet we know what's been shown.
    emitted_on_line: usize,
}

impl MarkdownStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of streamed text and print it live. Plain prose appears
    /// character-by-character (the Claude/GPT "typing" feel); inline spans
    /// (`**bold**`, `` `code` ``, `$math$`, `\commands`) are briefly held until
    /// they close, then printed converted. Newlines flush the line.
    pub fn push(&mut self, text: &str) {
        for ch in text.chars() {
            self.feed_char(ch);
        }
        let _ = io::stdout().flush();
    }

    /// Flush any buffered text at end of response.
    pub fn finish(&mut self) {
        self.flush_line(false);
        self.in_code_block = false;
        let _ = io::stdout().flush();
    }

    /// Process one character of the stream.
    fn feed_char(&mut self, ch: char) {
        if ch == '\r' {
            return; // normalize CRLF
        }
        if ch == '\n' {
            self.flush_line(true);
            return;
        }
        self.line_buf.push(ch);

        // Inside a fenced code block or a display-math block, don't stream live —
        // those are rendered whole at line/block boundaries (handled in
        // flush_line). Just keep buffering.
        if self.in_code_block || self.in_math_block {
            return;
        }

        // Try to emit as much of the line as is now "safe" to show — i.e. up to
        // the start of any still-open inline span. This is what makes prose
        // stream live while spans wait for their closing marker.
        self.try_stream_partial();
    }

    /// Stream the portion of the current line that's safe to print now: the
    /// block prefix (once detectable) plus all complete inline spans, leaving
    /// only an unclosed trailing span buffered.
    fn try_stream_partial(&mut self) {
        // Don't start streaming a line until we can tell whether it's a block
        // element (heading / list / quote / fence). A block prefix always lives
        // in the first few chars and ends at a space, so we buffer only that
        // short prefix region; once a space appears (or the line is clearly not
        // a prefix) we emit the prefix and stream the rest live.
        if !self.line_started {
            let trimmed = self.line_buf.trim_start();
            let has_space = trimmed.contains(' ');
            // Still ambiguous (could become "###", "- ", "1.") — wait for more,
            // unless it's a fence opener which we detect immediately.
            if !has_space && trimmed.len() < 4 && !trimmed.starts_with("``") {
                return;
            }
            // Headings, block quotes, fenced blocks, and display-math lines style
            // the *whole* line, so we don't stream them — they're buffered and
            // rendered in full at line flush. Everything else (prose, bullets,
            // numbered lists) streams live below.
            if trimmed.starts_with('#')
                || trimmed.starts_with("> ")
                || trimmed.starts_with("```")
                || trimmed.starts_with("\\[")
                || trimmed.starts_with("$$")
            {
                return;
            }
            // Emit the block prefix (bullet, number) now; stream the rest live.
            let (prefix, consumed) = block_prefix(&self.line_buf);
            print!("{prefix}");
            self.emitted_on_line = consumed;
            self.line_started = true;
        }

        // Stream complete inline spans from `emitted_on_line` up to the start of
        // any unclosed span.
        let safe_upto = safe_inline_boundary(&self.line_buf, self.emitted_on_line);
        if safe_upto > self.emitted_on_line {
            let chunk = &self.line_buf[self.emitted_on_line..safe_upto];
            print!("{}", render_inline_fragment(chunk));
            self.emitted_on_line = safe_upto;
        }
    }

    /// Emit whatever remains of the current line and reset line state. If
    /// `newline` is true, terminate with a newline.
    fn flush_line(&mut self, newline: bool) {
        let line = std::mem::take(&mut self.line_buf);
        let trimmed = line.trim_start();

        // --- Multi-line display-math block: \[ ... \]  or  $$ ... $$ ---
        if self.in_math_block {
            // Closing delimiter on its own line ends the block; render it all.
            if trimmed == "\\]" || trimmed == "$$" {
                let rendered = latex_to_unicode(self.math_buf.trim());
                println!("    {CYAN}{}{RESET}", rendered);
                self.math_buf.clear();
                self.in_math_block = false;
            } else {
                self.math_buf.push_str(&line);
                self.math_buf.push(' ');
            }
            self.reset_line_state();
            return;
        }
        // Opening a display-math block (delimiter alone on the line).
        if trimmed == "\\[" || trimmed == "$$" {
            self.in_math_block = true;
            self.math_buf.clear();
            self.reset_line_state();
            return;
        }
        // Single-line display math: \[ ... \] all on one line.
        if let Some(inner) = trimmed
            .strip_prefix("\\[")
            .and_then(|r| r.strip_suffix("\\]"))
        {
            println!("    {CYAN}{}{RESET}", latex_to_unicode(inner.trim()));
            self.reset_line_state();
            return;
        }

        // Fenced code block toggle.
        if trimmed.starts_with("```") {
            self.in_code_block = !self.in_code_block;
            let lang = trimmed.trim_start_matches('`');
            if self.in_code_block && !lang.is_empty() {
                println!("{DIM}┄┄ {lang} ┄┄{RESET}");
            } else {
                println!("{DIM}┄┄┄┄┄┄{RESET}");
            }
            self.reset_line_state();
            return;
        }

        if self.in_code_block {
            println!("{DIM}│{RESET} {line}");
            self.reset_line_state();
            return;
        }

        if !self.line_started {
            // Whole line buffered without streaming (short line): render fully.
            print!("{}", render_line(&line));
        } else {
            // Prefix + most spans already streamed; render the remaining tail.
            let tail = &line[self.emitted_on_line.min(line.len())..];
            print!("{}", render_inline_fragment(tail));
        }
        if newline {
            println!();
        }
        self.reset_line_state();
    }

    fn reset_line_state(&mut self) {
        self.line_started = false;
        self.emitted_on_line = 0;
    }
}

/// Detect a *streamable* block prefix (bullet or numbered list) at the start of
/// `line` and return its styled rendering plus bytes consumed. Headings, quotes
/// and code fences are handled separately (whole-line render), so they're not
/// here. Returns `("", 0)` for plain prose so it streams from the start.
fn block_prefix(line: &str) -> (String, usize) {
    let trimmed = line.trim_start();
    let indent_len = line.len() - trimmed.len();
    let indent = &line[..indent_len];

    for marker in ["- ", "* ", "+ "] {
        if trimmed.starts_with(marker) {
            return (format!("{indent}{GREEN}•{RESET} "), indent_len + marker.len());
        }
    }
    if let Some((num, _)) = split_numbered_list(trimmed) {
        return (
            format!("{indent}{GREEN}{num}.{RESET} "),
            indent_len + num.len() + 2,
        );
    }
    (String::new(), 0)
}

/// Given the current line and the byte offset already emitted, return the byte
/// offset up to which it's safe to render *now* — i.e. just before the start of
/// any still-unclosed inline span (`` ` ``, `**`, `*`/`_`, `$`, or a `\command`
/// that may still be growing). Returns the offset of the first unclosed marker,
/// or the full length if everything is closed.
fn safe_inline_boundary(line: &str, from: usize) -> usize {
    let bytes = line.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            // Inline code / math / emphasis: if the span isn't closed yet, stop
            // here and wait. If it is closed, it's safe to include — skip past.
            b'`' => match find_closing(line, i + 1, '`') {
                Some(end) => i = end + 1,
                None => return i,
            },
            b'$' => match line[i + 1..].find('$') {
                Some(rel) => i = i + 1 + rel + 1,
                None => return i,
            },
            b'*' => {
                // Bold ** or italic *
                if line[i..].starts_with("**") {
                    match line[i + 2..].find("**") {
                        Some(rel) => i = i + 2 + rel + 2,
                        None => return i,
                    }
                } else {
                    match find_closing(line, i + 1, '*') {
                        Some(end) => i = end + 1,
                        None => return i,
                    }
                }
            }
            b'_' => match find_closing(line, i + 1, '_') {
                Some(end) => i = end + 1,
                None => return i,
            },
            b'\\' => {
                let rest = &line[i..];
                // Inline math span `\( ... \)` — hold the whole span until it
                // closes, so multi-part expressions like \frac{a}{b} aren't split
                // across render calls (which would corrupt the conversion).
                if rest.starts_with("\\(") {
                    match rest.find("\\)") {
                        Some(rel) => i += rel + 2,
                        None => return i, // span still open — wait
                    }
                } else {
                    // A bare LaTeX command (e.g. \nabla, \frac). It may still be
                    // streaming in; hold until whitespace or line end confirms it
                    // and its braced args (if any) are complete.
                    let after = &rest[1..];
                    let cmd_end = after
                        .find(|c: char| c.is_whitespace() || c == '\\')
                        .map(|p| p + 1)
                        .unwrap_or(rest.len());
                    // If the command is immediately followed by `{`, it has args
                    // that must close before we render. Hold until the last brace.
                    let tail = &rest[cmd_end..];
                    if tail.starts_with('{') {
                        // Wait until all brace groups after the command are closed.
                        if brace_groups_closed(tail) {
                            i += utf8_char_len(bytes[i]); // safe; keep scanning
                        } else {
                            return i; // args still streaming
                        }
                    } else if rest[cmd_end..].is_empty() {
                        return i; // command name still growing at line end
                    } else {
                        i += utf8_char_len(bytes[i]);
                    }
                }
            }
            _ => i += utf8_char_len(bytes[i]),
        }
    }
    line.len()
}

/// Starting at a `{`, return true once the run of consecutive `{...}` groups is
/// fully balanced (so a command like `\frac{a}{b}` is complete). Returns false
/// if a brace group is still open — meaning we should wait for more stream.
fn brace_groups_closed(s: &str) -> bool {
    let mut depth = 0i32;
    let mut seen_open = false;
    for c in s.chars() {
        match c {
            '{' => {
                depth += 1;
                seen_open = true;
            }
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return true; // stray close; treat as done
                }
            }
            // Once balanced and we hit a non-brace, the group run is complete.
            _ if seen_open && depth == 0 => return true,
            _ => {}
        }
    }
    seen_open && depth == 0
}

/// Render a fragment of inline text (no block prefix) — applies LaTeX→Unicode
/// and emphasis/code styling. Used for live-streamed chunks within a line.
fn render_inline_fragment(s: &str) -> String {
    inline(s)
}

/// Render a single Markdown line (not inside a code block) to ANSI + Unicode.
/// Order: block-level prefix (heading / list / quote) → inline spans → LaTeX.
pub fn render_line(line: &str) -> String {
    let trimmed_start = line.trim_start();

    // Headings: #, ##, ### ...
    if let Some(rest) = trimmed_start.strip_prefix("#### ") {
        return format!("{BOLD}{CYAN}{}{RESET}", inline(rest));
    }
    if let Some(rest) = trimmed_start.strip_prefix("### ") {
        return format!("{BOLD}{CYAN}{}{RESET}", inline(rest));
    }
    if let Some(rest) = trimmed_start.strip_prefix("## ") {
        return format!("{BOLD}{CYAN}▌ {}{RESET}", inline(rest));
    }
    if let Some(rest) = trimmed_start.strip_prefix("# ") {
        return format!("{BOLD}{MAGENTA}▌ {}{RESET}", inline(rest));
    }

    // Horizontal rule.
    if trimmed_start == "---" || trimmed_start == "***" || trimmed_start == "___" {
        return format!("{DIM}────────────────────{RESET}");
    }

    // Block quote.
    if let Some(rest) = trimmed_start.strip_prefix("> ") {
        return format!("{DIM}│{RESET} {ITALIC}{}{RESET}", inline(rest));
    }

    // Bullet lists: -, *, + → •  (preserve leading indentation).
    let indent_len = line.len() - trimmed_start.len();
    let indent = &line[..indent_len];
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed_start.strip_prefix(marker) {
            return format!("{indent}{GREEN}•{RESET} {}", inline(rest));
        }
    }
    // Numbered lists: "1. ", "2. " ... keep the number, style the dot.
    if let Some((num, rest)) = split_numbered_list(trimmed_start) {
        return format!("{indent}{GREEN}{num}.{RESET} {}", inline(rest));
    }

    inline(line)
}

/// If `s` looks like "N. text", return (N, text). Used to detect ordered lists.
fn split_numbered_list(s: &str) -> Option<(&str, &str)> {
    let dot = s.find(". ")?;
    let (num, rest) = s.split_at(dot);
    if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) {
        Some((num, &rest[2..]))
    } else {
        None
    }
}

/// Render inline Markdown spans (bold, italic, inline code) and inline LaTeX
/// math within a single line. Applied after block-level handling.
fn inline(s: &str) -> String {
    // LaTeX first so `$...$` content doesn't get mistaken for emphasis.
    let s = render_latex(s);
    render_spans(&s)
}

/// Replace `**bold**`, `*italic*` / `_italic_`, and `` `code` `` with ANSI.
/// Single forward pass; unmatched markers are left as literal text.
///
/// UTF-8-safe: it indexes by byte (the markers `*` `_` `` ` `` are all ASCII, so
/// every index we slice at is a valid char boundary) but copies whole byte
/// *ranges* of non-marker text via `&s[a..b]`, never casting individual bytes to
/// `char`. (Casting a single byte of a multi-byte char like ∇ produces garbage —
/// that was the source of the `âÂ²` corruption.)
fn render_spans(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let bytes = s.as_bytes();
    let mut i = 0;
    // Start of the current run of plain (non-marker) text to be copied verbatim.
    let mut plain_start = 0;

    // Copy `s[plain_start..i]` to `out`, then reset the run start to `new_start`.
    macro_rules! flush_plain {
        ($out:expr, $new_start:expr) => {{
            if plain_start < i {
                $out.push_str(&s[plain_start..i]);
            }
            plain_start = $new_start;
        }};
    }

    while i < bytes.len() {
        // Inline code: `...`
        if bytes[i] == b'`' {
            if let Some(end) = find_closing(s, i + 1, '`') {
                flush_plain!(out, end + 1);
                out.push_str(YELLOW);
                out.push_str(&s[i + 1..end]);
                out.push_str(RESET);
                i = end + 1;
                continue;
            }
        }
        // Bold: **...**
        if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            if let Some(rel) = s[i + 2..].find("**") {
                let end = i + 2 + rel;
                flush_plain!(out, end + 2);
                out.push_str(BOLD);
                out.push_str(&s[i + 2..end]);
                out.push_str(RESET);
                i = end + 2;
                continue;
            }
        }
        // Italic: *...*  or _..._  (single marker)
        if bytes[i] == b'*' || bytes[i] == b'_' {
            let marker = bytes[i] as char; // marker is ASCII, safe
            if let Some(end) = find_closing(s, i + 1, marker) {
                // Avoid treating mid-word underscores (a_b) as italics.
                let inner = &s[i + 1..end];
                if !inner.is_empty() && !inner.contains(marker) {
                    flush_plain!(out, end + 1);
                    out.push_str(ITALIC);
                    out.push_str(inner);
                    out.push_str(RESET);
                    i = end + 1;
                    continue;
                }
            }
        }
        // Advance by a full UTF-8 char so `i` always lands on a char boundary.
        i += utf8_char_len(bytes[i]);
    }
    // Copy the trailing plain run.
    if plain_start < bytes.len() {
        out.push_str(&s[plain_start..]);
    }
    out
}

/// Length in bytes of the UTF-8 char that starts with `first_byte`. Lets us step
/// a byte index forward one whole character at a time.
fn utf8_char_len(first_byte: u8) -> usize {
    match first_byte {
        b if b < 0x80 => 1,
        b if b >> 5 == 0b110 => 2,
        b if b >> 4 == 0b1110 => 3,
        b if b >> 3 == 0b11110 => 4,
        _ => 1, // invalid lead byte; advance 1 to avoid stalling
    }
}

/// Find the next unescaped `marker` char at/after `from`, returning its index.
fn find_closing(s: &str, from: usize, marker: char) -> Option<usize> {
    s[from..].find(marker).map(|rel| from + rel)
}

/// Convert common inline LaTeX math into readable Unicode. Handles both `$...$`
/// and `\( ... \)` delimiters, plus a handful of bare commands that models emit
/// outside math mode. This is best-effort prettification, not a TeX engine:
///   * superscripts:  x^2 → x²,  x^{10} → x¹⁰
///   * subscripts:    a_1 → a₁,  H_{2}O → H₂O
///   * Greek letters: \alpha → α, \beta → β, ...
///   * operators:     \times → ×, \cdot → ·, \leq → ≤, \frac{a}{b} → (a)/(b)
pub fn render_latex(s: &str) -> String {
    // Single left-to-right scan. Each region of the input is converted exactly
    // once — delimited math spans (`$...$`, `\(...\)`) get styled+converted, and
    // the plain text *between* them gets a bare-command conversion (so models'
    // out-of-math `\nabla`, `\frac{a}{b}`, `x^2` still render). Doing this in one
    // pass avoids re-processing already-converted spans (which previously mangled
    // braces and ANSI codes).
    let mut out = String::with_capacity(s.len() + 16);
    let mut rest = s;
    // Delimiter pairs, longest-open-first so `$$` wins over `$`.
    let delims: &[(&str, &str)] = &[("$$", "$$"), ("$", "$"), ("\\(", "\\)")];

    'outer: loop {
        // Find the earliest delimiter opening in `rest`.
        let mut best: Option<(usize, &str, &str)> = None;
        for (open, close) in delims {
            if let Some(pos) = rest.find(open) {
                if best.map(|(b, _, _)| pos < b).unwrap_or(true) {
                    best = Some((pos, open, close));
                }
            }
        }

        let Some((start, open, close)) = best else {
            // No more math spans — convert the trailing plain region and finish.
            out.push_str(&convert_bare(rest));
            break 'outer;
        };

        // Plain text before the span: bare-command conversion.
        out.push_str(&convert_bare(&rest[..start]));

        let after_open = &rest[start + open.len()..];
        let Some(end_rel) = after_open.find(close) else {
            // Unterminated span — treat the rest as plain.
            out.push_str(&convert_bare(&rest[start..]));
            break 'outer;
        };
        let inner = &after_open[..end_rel];
        out.push_str(&format!("{CYAN}{}{RESET}", latex_to_unicode(inner)));
        rest = &after_open[end_rel + close.len()..];
    }
    out
}

/// Convert a plain (non-delimited) text region: apply LaTeX→Unicode only if it
/// actually contains a backslash command, so ordinary prose is left untouched.
fn convert_bare(s: &str) -> String {
    if s.contains('\\') {
        latex_to_unicode(s)
    } else {
        s.to_string()
    }
}

/// Translate a LaTeX math fragment into Unicode. Applied to the inside of a
/// math span. Conservative: anything it doesn't recognize passes through.
pub fn latex_to_unicode(s: &str) -> String {
    let mut s = s.to_string();

    // Strip wrapper commands that only style their braced content — keep the
    // content, drop the command + braces. e.g. \boxed{x}, \text{ m }, \mathbf{v}.
    for cmd in ["\\boxed", "\\text", "\\mathrm", "\\mathbf", "\\mathit", "\\operatorname"] {
        s = strip_wrapper(&s, cmd);
    }

    // \frac{a}{b} → (a)/(b)
    s = replace_frac(&s);

    // Commands → symbols. Longer names first to avoid partial shadowing.
    const SYMBOLS: &[(&str, &str)] = &[
        ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
        ("\\epsilon", "ε"), ("\\varepsilon", "ε"), ("\\zeta", "ζ"), ("\\eta", "η"),
        ("\\theta", "θ"), ("\\iota", "ι"), ("\\kappa", "κ"), ("\\lambda", "λ"),
        ("\\mu", "μ"), ("\\nu", "ν"), ("\\xi", "ξ"), ("\\pi", "π"),
        ("\\rho", "ρ"), ("\\sigma", "σ"), ("\\tau", "τ"), ("\\phi", "φ"),
        ("\\varphi", "φ"), ("\\chi", "χ"), ("\\psi", "ψ"), ("\\omega", "ω"),
        ("\\Gamma", "Γ"), ("\\Delta", "Δ"), ("\\Theta", "Θ"), ("\\Lambda", "Λ"),
        ("\\Pi", "Π"), ("\\Sigma", "Σ"), ("\\Phi", "Φ"), ("\\Psi", "Ψ"),
        ("\\Omega", "Ω"),
        ("\\times", "×"), ("\\cdot", "·"), ("\\div", "÷"), ("\\pm", "±"),
        ("\\leq", "≤"), ("\\geq", "≥"), ("\\neq", "≠"), ("\\approx", "≈"),
        ("\\equiv", "≡"), ("\\propto", "∝"), ("\\infty", "∞"),
        ("\\rightarrow", "→"), ("\\to", "→"), ("\\leftarrow", "←"),
        ("\\Rightarrow", "⇒"), ("\\Leftarrow", "⇐"), ("\\leftrightarrow", "↔"),
        ("\\sum", "∑"), ("\\prod", "∏"), ("\\int", "∫"), ("\\partial", "∂"),
        ("\\nabla", "∇"), ("\\sqrt", "√"), ("\\forall", "∀"), ("\\exists", "∃"),
        ("\\in", "∈"), ("\\notin", "∉"), ("\\subset", "⊂"), ("\\cup", "∪"),
        ("\\cap", "∩"), ("\\emptyset", "∅"), ("\\angle", "∠"), ("\\degree", "°"),
        ("\\ldots", "…"), ("\\cdots", "⋯"), ("\\left", ""), ("\\right", ""),
        ("\\,", " "), ("\\;", " "), ("\\!", ""), ("\\quad", "  "),
    ];
    for (tex, uni) in SYMBOLS {
        if s.contains(tex) {
            s = s.replace(tex, uni);
        }
    }

    // Superscripts and subscripts.
    s = render_scripts(&s, '^', SUPERSCRIPTS);
    s = render_scripts(&s, '_', SUBSCRIPTS);

    // Drop stray braces left over from {...} groups we didn't otherwise consume.
    s.replace(['{', '}'], "")
}

/// Remove a styling wrapper command, keeping its braced argument. e.g. with
/// `cmd = "\\boxed"`: `\boxed{x^2}` → `x^2`. Leaves text alone if the command
/// isn't immediately followed by a `{...}` group.
fn strip_wrapper(s: &str, cmd: &str) -> String {
    let needle = format!("{cmd}{{");
    let mut out = s.to_string();
    while let Some(pos) = out.find(&needle) {
        let after = &out[pos + needle.len()..];
        let Some((inner, rest)) = take_braced_inner(after) else { break };
        out = format!("{}{}{}", &out[..pos], inner, rest);
    }
    out
}

/// `\frac{a}{b}` → `(a)/(b)`. Handles repeated occurrences; nested fractions
/// resolve outermost-first across passes (good enough for typical model output).
fn replace_frac(s: &str) -> String {
    let mut out = s.to_string();
    while let Some(pos) = out.find("\\frac{") {
        // `pos + 6` lands just past `\frac{`, i.e. at the numerator content.
        let after = &out[pos + 6..];
        let Some((num, rest1)) = take_braced_inner(after) else { break };
        // The denominator must immediately follow as `{...}`.
        let rest1 = rest1.trim_start();
        let Some(rest1) = rest1.strip_prefix('{') else { break };
        let Some((den, rest2)) = take_braced_inner(rest1) else { break };
        let replacement = format!("({})/({})", num, den);
        out = format!("{}{}{}", &out[..pos], replacement, rest2);
    }
    out
}

/// Given text right after a `{`, return (inner, remainder-after-closing-brace).
fn take_braced_inner(s: &str) -> Option<(String, &str)> {
    let mut depth = 1;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((s[..i].to_string(), &s[i + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

/// Replace `base^x` / `base_x` runs with Unicode super/subscripts where every
/// character maps; otherwise leaves a readable `^x` / `_x` fallback. Supports
/// braced groups: `x^{10}` and bare single chars: `x^2`.
fn render_scripts(s: &str, marker: char, table: &[(char, char)]) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == marker && i + 1 < chars.len() {
            // Collect the script content: a {group} or a single char.
            let (content, next_i) = if chars[i + 1] == '{' {
                let mut j = i + 2;
                let mut buf = String::new();
                while j < chars.len() && chars[j] != '}' {
                    buf.push(chars[j]);
                    j += 1;
                }
                (buf, j + 1) // skip past '}'
            } else {
                (chars[i + 1].to_string(), i + 2)
            };

            // Map each char; fall back to literal `^content` if any char is
            // unmappable, so we never produce confusing partial output.
            let mapped: Option<String> = content
                .chars()
                .map(|c| table.iter().find(|(k, _)| *k == c).map(|(_, v)| *v))
                .collect();
            match mapped {
                Some(m) if !m.is_empty() => out.push_str(&m),
                _ => {
                    out.push(marker);
                    out.push_str(&content);
                }
            }
            i = next_i;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Digit/letter → superscript Unicode.
const SUPERSCRIPTS: &[(char, char)] = &[
    ('0', '⁰'), ('1', '¹'), ('2', '²'), ('3', '³'), ('4', '⁴'),
    ('5', '⁵'), ('6', '⁶'), ('7', '⁷'), ('8', '⁸'), ('9', '⁹'),
    ('+', '⁺'), ('-', '⁻'), ('=', '⁼'), ('(', '⁽'), (')', '⁾'),
    ('n', 'ⁿ'), ('i', 'ⁱ'),
];

/// Digit → subscript Unicode.
const SUBSCRIPTS: &[(char, char)] = &[
    ('0', '₀'), ('1', '₁'), ('2', '₂'), ('3', '₃'), ('4', '₄'),
    ('5', '₅'), ('6', '₆'), ('7', '₇'), ('8', '₈'), ('9', '₉'),
    ('+', '₊'), ('-', '₋'), ('=', '₌'), ('(', '₍'), (')', '₎'),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(s: &str) -> String {
        // Crude ANSI stripper for assertions.
        let mut out = String::new();
        let mut in_esc = false;
        for c in s.chars() {
            if in_esc {
                if c == 'm' {
                    in_esc = false;
                }
            } else if c == '\x1b' {
                in_esc = true;
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn superscript_and_subscript() {
        assert_eq!(latex_to_unicode("x^2"), "x²");
        assert_eq!(latex_to_unicode("x^{10}"), "x¹⁰");
        assert_eq!(latex_to_unicode("H_2O"), "H₂O");
    }

    #[test]
    fn greek_and_operators() {
        assert_eq!(latex_to_unicode("\\alpha + \\beta"), "α + β");
        assert_eq!(latex_to_unicode("a \\times b"), "a × b");
        assert_eq!(latex_to_unicode("x \\leq y"), "x ≤ y");
    }

    #[test]
    fn fraction() {
        assert_eq!(latex_to_unicode("\\frac{a}{b}"), "(a)/(b)");
        assert_eq!(latex_to_unicode("\\frac{1}{2}"), "(1)/(2)");
    }

    #[test]
    fn boxed_and_text_wrappers_unwrap() {
        assert_eq!(latex_to_unicode("\\boxed{a^2 + b^2 = c^2}"), "a² + b² = c²");
        assert_eq!(latex_to_unicode("5 \\text{ m}"), "5  m");
    }

    #[test]
    fn frac_with_commands_inside() {
        // The real-world failing case: \frac with \partial in numerator/denominator.
        assert_eq!(
            latex_to_unicode("\\frac{\\partial u}{\\partial x}"),
            "(∂ u)/(∂ x)"
        );
    }

    #[test]
    fn inline_math_delimiters_stripped() {
        // The $...$ delimiters are removed; superscript digits convert.
        let r = strip_ansi(&render_latex("area is $r^2$ units"));
        assert_eq!(r, "area is r² units");
    }

    #[test]
    fn bold_and_code_spans() {
        let r = strip_ansi(&render_line("This is **bold** and `code`"));
        assert_eq!(r, "This is bold and code");
    }

    #[test]
    fn heading_renders_text() {
        let r = strip_ansi(&render_line("## Section"));
        assert!(r.contains("Section"));
    }

    #[test]
    fn bullet_becomes_dot() {
        let r = strip_ansi(&render_line("- item"));
        assert!(r.contains("item"));
        assert!(r.contains('•'));
    }
}
