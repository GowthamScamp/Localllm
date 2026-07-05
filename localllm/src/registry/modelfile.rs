//! Lightweight Modelfile parser, Ollama-compatible.
//!
//! Supported directives:
//!   FROM <model_alias>         — base model (required for `create`)
//!   SYSTEM "system prompt"     — system message prepended to all chats
//!   TEMPLATE """{{ .System }}{{ .Prompt }}"""  — Go-template style render
//!   PARAMETER <key> <value>    — inference parameter (temperature, top_p, ...)
//!   MESSAGE <role> "<content>" — seed conversation history
//!
//! Anything else is ignored with a warning.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Parsed representation of an Ollama-style Modelfile. Stored on the manifest
/// (`ModelManifest.modelfile`) and applied at request time to inject SYSTEM
/// messages, seed conversation history, and provide PARAMETER defaults.
/// The original `source` text is kept so `/api/show` can round-trip it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Modelfile {
    /// Base model alias this Modelfile derives from (e.g. "llama-3.2-1b-instruct").
    pub from: Option<String>,
    /// SYSTEM message to prepend.
    pub system: Option<String>,
    /// TEMPLATE string (Go template syntax; we apply minimally on chat input).
    pub template: Option<String>,
    /// PARAMETER key/value pairs.
    pub parameters: Vec<(String, String)>,
    /// MESSAGE entries used as seed history.
    pub messages: Vec<TemplateMessage>,
    /// Original source text (for /api/show round-tripping).
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

impl Modelfile {
    /// Parse a Modelfile source string.
    ///
    /// Tokenization: directive (uppercased) + value. Value can be a single
    /// word, a quoted string, or a triple-quoted multi-line block. Comments
    /// (`#`) and blank lines are ignored. Unknown directives log a warning
    /// and are dropped — keeps us forward-compat with new Ollama directives.
    ///
    /// Returns Err only for unterminated triple-quoted blocks; everything
    /// else degrades gracefully.
    pub fn parse(source: &str) -> Result<Self> {
        let mut mf = Modelfile {
            source: source.to_string(),
            ..Default::default()
        };

        // Track multi-line triple-quoted blocks
        let mut current_directive: Option<String> = None;
        let mut current_buffer = String::new();
        let mut in_triple_quote = false;

        for raw_line in source.lines() {
            let line = raw_line.trim_end();

            if in_triple_quote {
                if let Some(end) = line.find("\"\"\"") {
                    current_buffer.push_str(&line[..end]);
                    in_triple_quote = false;
                    let value = std::mem::take(&mut current_buffer);
                    if let Some(d) = current_directive.take() {
                        mf.apply(&d, value.trim());
                    }
                } else {
                    current_buffer.push_str(line);
                    current_buffer.push('\n');
                }
                continue;
            }

            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Split into directive + rest
            let (directive, rest) = match trimmed.split_once(char::is_whitespace) {
                Some((d, r)) => (d.to_uppercase(), r.trim()),
                None => (trimmed.to_uppercase(), ""),
            };

            // Triple-quoted multiline value?
            if let Some(after_open) = rest.strip_prefix("\"\"\"") {
                if let Some(end) = after_open.find("\"\"\"") {
                    // Single-line triple-quoted: """value"""
                    mf.apply(&directive, after_open[..end].trim());
                } else {
                    in_triple_quote = true;
                    current_directive = Some(directive);
                    current_buffer.clear();
                    current_buffer.push_str(after_open);
                    current_buffer.push('\n');
                }
                continue;
            }

            // Plain single-line value, with optional surrounding quotes
            let value = strip_quotes(rest);
            mf.apply(&directive, value);
        }

        if in_triple_quote {
            return Err(anyhow!("Unterminated triple-quoted block in Modelfile"));
        }

        Ok(mf)
    }

    /// Dispatch one directive into the appropriate field. Called by the parser
    /// once per logical line. Malformed `PARAMETER`/`MESSAGE` lines (missing
    /// the required second token) log a warning and are dropped.
    fn apply(&mut self, directive: &str, value: &str) {
        match directive {
            "FROM" => self.from = Some(value.to_string()),
            "SYSTEM" => self.system = Some(value.to_string()),
            "TEMPLATE" => self.template = Some(value.to_string()),
            "PARAMETER" => {
                if let Some((k, v)) = value.split_once(char::is_whitespace) {
                    self.parameters
                        .push((k.trim().to_lowercase(), v.trim().to_string()));
                } else {
                    tracing::warn!("Malformed PARAMETER (need key + value): {}", value);
                }
            }
            "MESSAGE" => {
                if let Some((role, content)) = value.split_once(char::is_whitespace) {
                    self.messages.push(TemplateMessage {
                        role: role.trim().to_lowercase(),
                        content: strip_quotes(content.trim()).to_string(),
                    });
                } else {
                    tracing::warn!("Malformed MESSAGE: {}", value);
                }
            }
            "LICENSE" | "ADAPTER" | "MIROSTAT" => {
                // Recognized Ollama directives we don't act on yet — silent ignore
            }
            other => {
                tracing::warn!("Unknown Modelfile directive: {}", other);
            }
        }
    }

    /// Render the TEMPLATE string with the given prompt + optional system text.
    /// Supports the three Ollama placeholders most templates use:
    ///   `{{ .System }}`, `{{ .Prompt }}`, `{{ .Response }}`
    ///
    /// Whitespace inside the braces is tolerated. If no TEMPLATE is set,
    /// returns the prompt unchanged — the safe fallback for `/api/generate`.
    pub fn render_template(&self, prompt: &str) -> String {
        let tpl = match &self.template {
            Some(t) => t.clone(),
            None => return prompt.to_string(),
        };
        let system = self.system.as_deref().unwrap_or("");
        substitute(&tpl, system, prompt, "")
    }

    /// Compose the final message list to send upstream: optional SYSTEM
    /// prompt + seeded `MESSAGE` history + caller's messages.
    ///
    /// Skip-rule: if the caller already supplied any `role: "system"` message,
    /// we DON'T add our SYSTEM (caller wins — they explicitly overrode). All
    /// SEED messages still get inserted before user messages either way.
    pub fn apply_to_messages(
        &self,
        user_messages: &[crate::api::types::ChatMessage],
    ) -> Vec<crate::api::types::ChatMessage> {
        let mut out: Vec<crate::api::types::ChatMessage> = Vec::with_capacity(
            user_messages.len() + self.messages.len() + 1,
        );

        let caller_has_system = user_messages.iter().any(|m| m.role == "system");

        if !caller_has_system {
            if let Some(sys) = &self.system {
                out.push(crate::api::types::ChatMessage {
                    role: "system".to_string(),
                    content: sys.clone(),
                });
            }
        }

        for seed in &self.messages {
            out.push(crate::api::types::ChatMessage {
                role: seed.role.clone(),
                content: seed.content.clone(),
            });
        }

        out.extend_from_slice(user_messages);
        out
    }
}

/// Strip a single pair of matching outer quotes (single or double).
/// Idempotent on already-unquoted strings. Used so users can write either
/// `SYSTEM "hello"` or `SYSTEM hello` and get the same result.
fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 {
        let bytes = s.as_bytes();
        if (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
        {
            return &s[1..s.len() - 1];
        }
    }
    s
}

/// Replace `{{ .Name }}` placeholders in `tpl` with the corresponding values.
/// Whitespace around the dotted name is tolerated; unknown names are left as-is
/// so a template that uses a placeholder we don't support yet still survives.
fn substitute(tpl: &str, system: &str, prompt: &str, response: &str) -> String {
    let mut out = String::with_capacity(tpl.len() + system.len() + prompt.len());
    // Index into the &str by byte offset, but only ever slice on char
    // boundaries (the `{{`/`}}` delimiters are ASCII, so the offsets we compute
    // are always boundaries). We copy the in-between text as UTF-8 string
    // slices rather than casting raw bytes to `char`, which would corrupt any
    // multi-byte character (e.g. non-ASCII text inside a TEMPLATE).
    let mut rest = tpl;
    while let Some(open) = rest.find("{{") {
        // Emit everything before the opening braces verbatim.
        out.push_str(&rest[..open]);
        let after_open = &rest[open + 2..];
        if let Some(close) = after_open.find("}}") {
            let inner = after_open[..close].trim();
            // Match `.Name` (case-insensitive on name).
            let key = inner.trim_start_matches('.').to_ascii_lowercase();
            let value = match key.as_str() {
                "system" => Some(system),
                "prompt" => Some(prompt),
                "response" => Some(response),
                _ => None,
            };
            match value {
                Some(v) => out.push_str(v),
                // Unknown placeholder — leave it untouched so unsupported
                // template vars survive verbatim.
                None => {
                    out.push_str("{{");
                    out.push_str(&after_open[..close]);
                    out.push_str("}}");
                }
            }
            rest = &after_open[close + 2..];
        } else {
            // No closing braces — emit the rest verbatim and stop.
            out.push_str("{{");
            out.push_str(after_open);
            return out;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_modelfile() {
        let src = r#"
FROM llama-3.2-1b-instruct
SYSTEM "You are a helpful assistant."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"#;
        let mf = Modelfile::parse(src).unwrap();
        assert_eq!(mf.from.as_deref(), Some("llama-3.2-1b-instruct"));
        assert_eq!(mf.system.as_deref(), Some("You are a helpful assistant."));
        assert_eq!(mf.parameters.len(), 2);
        assert_eq!(mf.parameters[0], ("temperature".to_string(), "0.7".to_string()));
    }

    #[test]
    fn substitute_preserves_multibyte_utf8() {
        // Regression: an earlier byte-by-byte implementation cast each byte to
        // `char`, corrupting any multi-byte UTF-8 in the template's literal text.
        let tpl = "Café ☕ {{ .Prompt }} — naïve";
        let out = substitute(tpl, "", "héllo 你好", "");
        assert_eq!(out, "Café ☕ héllo 你好 — naïve");
    }

    #[test]
    fn substitute_leaves_unknown_placeholders() {
        let out = substitute("{{ .Prompt }} {{ .Unknown }}", "", "hi", "");
        assert_eq!(out, "hi {{ .Unknown }}");
    }

    #[test]
    fn parses_triple_quoted_template() {
        let src = r#"
FROM base
TEMPLATE """{{ .System }}
User: {{ .Prompt }}
Assistant:"""
"#;
        let mf = Modelfile::parse(src).unwrap();
        let tpl = mf.template.unwrap();
        assert!(tpl.contains("{{ .Prompt }}"));
    }
}
