//! # CLI command dispatcher
//!
//! Every subcommand here is a thin HTTP client over the daemon's REST API.
//! The CLI never touches model files, registry, or inference processes
//! directly — it only sends JSON requests and pretty-prints the responses.
//! This keeps the binary single-purpose and means the daemon stays the single
//! source of truth for state (file locks, port allocations, manifest cache).
//!
//! Pattern for every command: build a request struct → POST/GET → check status
//! → format response to stdout. Errors bubble up as `anyhow::Error` and the
//! framework prints them to stderr with a non-zero exit code.

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::time::Duration;

use crate::api::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, LoadRequest, OllamaCopyRequest,
    OllamaCreateRequest, OllamaShowRequest, QuantizeRequest,
};
use crate::config::Settings;

/// System prompt seeded into every interactive chat. Primes the model to behave
/// like a capable general assistant with clean, terminal-friendly formatting.
const SYSTEM_PROMPT: &str = "You are a helpful, knowledgeable AI assistant. \
Answer clearly and concisely. Use Markdown for structure (headings, bullet \
points, **bold**, and `code`) when it improves readability. Write code inside \
fenced ``` blocks with a language tag. For mathematics, use LaTeX: inline math \
in \\( ... \\) and display math in \\[ ... \\]. Be direct and get to the point.";

/// Chat sampling defaults — tuned for natural, coherent replies, comparable to
/// hosted assistant chat UIs.
const CHAT_TEMPERATURE: f32 = 0.7;
const CHAT_TOP_P: f32 = 0.9;
const CHAT_MAX_TOKENS: u32 = 2048;

/// Top-level CLI parser. `clap` derives the arg-parsing logic from the field
/// attributes — running `localllm --help` prints the auto-generated help.
#[derive(Parser)]
#[command(
    name = "localllm",
    version,
    about = "Production-grade local LLM serving tool"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Override the daemon URL (default: built from config `daemon_host:port`).
    /// Also reads from `LOCALLLM_DAEMON_URL` env if not passed on the command line.
    #[arg(long, env = "LOCALLLM_DAEMON_URL")]
    pub daemon_url: Option<String>,

    /// HuggingFace access token, forwarded to the daemon on pulls. Enables
    /// gated/private models and raises HF rate limits (faster downloads).
    /// Falls back to the `HF_TOKEN` environment variable.
    #[arg(long, env = "HF_TOKEN", hide_env_values = true, global = true)]
    pub hf_token: Option<String>,
}

/// All localllm subcommands. Each variant maps 1:1 to a CLI verb.
#[derive(Subcommand)]
pub enum Commands {
    /// Run the long-lived HTTP daemon in this process (no auto-spawn).
    Serve,
    /// Download a HuggingFace model into the local registry. Optionally
    /// quantize to GGUF in one step (requires `llama_cpp_dir` configured).
    Pull {
        repo_id: String,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long)]
        quantize: Option<String>,
    },
    /// Send a chat completion to a model. Single-shot if `prompt` is given,
    /// REPL loop if `--interactive`. Tokens stream to stdout as they arrive.
    Run {
        model: String,
        prompt: Option<String>,
        #[arg(short = 'i', long)]
        interactive: bool,
    },
    /// Quantize an already-pulled model to a smaller GGUF format
    /// (Q4_K_M default — best balance of size and quality).
    Quantize {
        model: String,
        #[arg(long, default_value = "Q4_K_M")]
        level: String,
    },
    /// Print a table of all registered models with status (loaded/ready).
    List,
    /// Print a table of currently-running inference processes.
    Ps,
    /// Eagerly warm a model into memory (instead of waiting for first request).
    Load {
        model: String,
    },
    /// Delete a model from the registry and disk. `--yes` skips confirmation.
    Rm {
        model: String,
        #[arg(long)]
        yes: bool,
    },
    /// Print detailed info about a model — template, parameters, modelfile.
    Show {
        model: String,
    },
    /// Alias a model under a new name without copying files.
    Cp {
        source: String,
        destination: String,
    },
    /// Create a derived model from a Modelfile (SYSTEM, PARAMETER, MESSAGE).
    Create {
        name: String,
        #[arg(short = 'f', long = "file")]
        file: String,
    },
    /// Tail recent stdout/stderr lines from the inference process for a model.
    /// Useful for debugging slow loads or backend crashes without enabling
    /// daemon-wide RUST_LOG=debug.
    Logs {
        model: String,
        #[arg(long, default_value_t = 100)]
        lines: usize,
    },
    /// **B1** — One-liner: pull (if needed) → quantize-if-CPU-only → interactive chat.
    /// Accepts either a HuggingFace repo (`org/Name`) or a local alias.
    Chat {
        target: String,
        /// Force a quantization level even if a GPU is available.
        #[arg(long)]
        quantize: Option<String>,
    },
    /// **B2** — Run diagnostics: GPU, sglang, llama.cpp, daemon, HF token, disk.
    /// Reports each subsystem's status and tells you what to fix.
    Doctor,
    /// **B4** — Print shell completion script (bash, zsh, fish, powershell).
    /// Source the output in your shell's rc file to enable tab-completion.
    Completion {
        shell: clap_complete::Shell,
    },
    /// **B5** — Tail daemon log stream (recent stdout/stderr from inference
    /// processes for all loaded models, interleaved).
    Tail {
        #[arg(long, default_value_t = 100)]
        lines: usize,
    },
    /// Build the inference engine (llama.cpp) now instead of on first use.
    /// Runs automatically on first run; this is here for users who want to
    /// pre-build it explicitly (e.g. right after `cargo build`). Pass
    /// `--rebuild` to force a fresh build — e.g. after installing a GPU toolkit
    /// (CUDA / Vulkan) so the engine is rebuilt with GPU acceleration.
    Setup {
        #[arg(long)]
        rebuild: bool,
    },
    /// Install the `localllm` binary onto your PATH so you can run it from
    /// anywhere (copies to ~/.local/bin on Unix / a user bin dir on Windows).
    Install,
}

/// Build a tuned reqwest client shared across every CLI invocation.
///
/// Key choice: **no overall timeout**. A streaming chat response can take
/// many minutes for long generations; an overall timeout would chop the
/// stream mid-token. Only `connect_timeout` (5s) is set, which fails fast
/// when the daemon is down without limiting in-flight responses.
///
/// `reqwest::Client` holds an `Arc` internally — cloning it is ~free and
/// shares the connection pool, so the cost of "build once, use anywhere"
/// is paid only at startup.
fn build_cli_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(4)
        .tcp_nodelay(true)
        // No overall timeout — chat streaming responses can take minutes.
        .connect_timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default()
}

/// Main dispatch — match on the subcommand and call the matching daemon endpoint.
///
/// `_settings` is currently unused but kept in the signature so future commands
/// that need to read config locally (e.g. show paths) don't break the API.
pub async fn execute(cli: Cli, _settings: Arc<Settings>, daemon_url: String) -> Result<()> {
    let client = build_cli_client(); // A1: built once, reused below
    // Captured before the match consumes `cli.command`; forwarded to every pull.
    let hf_token = cli.hf_token.clone();

    match cli.command {
        Commands::Serve => unreachable!("Serve is handled in main"),

        Commands::Pull {
            repo_id,
            revision,
            quantize,
        } => {
            println!("Pulling {}...", repo_id);
            stream_pull(&client, &daemon_url, &repo_id, revision.as_deref(), quantize.as_deref(), hf_token.as_deref()).await?;
        }

        Commands::Run {
            model,
            prompt,
            interactive,
        } => {
            // If no GPU is available, check whether the model already has a
            // GGUF on disk. Skip the pull if it does; otherwise auto-pull a
            // prebuilt Q4_K_M GGUF (same logic as `localllm chat`).
            if !has_local_gpu() {
                let models_resp = client
                    .get(format!("{}/api/models", daemon_url))
                    .send()
                    .await?;
                if models_resp.status().is_success() {
                    let models: Vec<serde_json::Value> =
                        models_resp.json().await.unwrap_or_default();
                    let manifest = models
                        .iter()
                        .find(|m| m["alias"].as_str() == Some(model.as_str()));
                    // A model is runnable on CPU if it already has a GGUF file.
                    // Prefer the explicit gguf_path; fall back to weight_format
                    // ("GGUF") for compatibility with older daemons that don't
                    // expose the path.
                    let has_gguf = |m: &serde_json::Value| {
                        m["gguf_path"].as_str().is_some()
                            || m["weight_format"].as_str() == Some("GGUF")
                    };
                    match manifest {
                        Some(m) if has_gguf(m) => {
                            // GGUF already on disk — nothing to download.
                            match m["gguf_path"].as_str() {
                                Some(p) => println!("Using cached GGUF: {}", p),
                                None => println!("Using cached GGUF model."),
                            }
                        }
                        Some(_) => {
                            println!("No GPU detected and model has no GGUF — pulling Q4_K_M quantized file...");
                            stream_pull(&client, &daemon_url, &model, None, Some("Q4_K_M"), hf_token.as_deref()).await?;
                        }
                        None => {}
                    }
                }
            }
            if interactive {
                run_interactive(&client, &daemon_url, &model).await?;
            } else {
                let prompt_text = prompt.ok_or_else(|| {
                    anyhow!("Prompt required (or use --interactive)")
                })?;
                run_single(&client, &daemon_url, &model, &prompt_text).await?;
            }
        }

        Commands::Quantize { model, level } => {
            println!("Quantizing {} to {}...", model, level);
            let resp = client
                .post(format!("{}/api/quantize", daemon_url))
                .json(&QuantizeRequest {
                    alias: model.clone(),
                    level,
                })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Quantize failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            let manifest: serde_json::Value = resp.json().await?;
            println!("Quantization complete:");
            println!("  alias:        {}", manifest["alias"].as_str().unwrap_or("-"));
            if let Some(q) = manifest["quantization"].as_str() {
                println!("  quantization: {}", q);
            }
            if let Some(p) = manifest["gguf_path"].as_str() {
                println!("  gguf_path:    {}", p);
            }
        }

        Commands::List => {
            let resp = client
                .get(format!("{}/api/models", daemon_url))
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!("List failed: {}", resp.text().await.unwrap_or_default()));
            }
            let models: Vec<serde_json::Value> = resp.json().await?;
            use crate::cli::style;
            println!(
                "{}",
                style::dim(&format!(
                    "{:<25} {:<40} {:<8} {:<10} {:<14} {:<8}",
                    "ALIAS", "REPO_ID", "PARAMS", "QUANT", "FORMAT", "STATUS"
                ))
            );
            for m in &models {
                let status = m["status"].as_str().unwrap_or("-");
                // Color the status: loaded=green, ready=cyan, anything else=dim.
                let status_colored = match status {
                    "loaded" => style::ok(status),
                    "ready" => format!("{}{}{}", style::CYAN, status, style::RESET),
                    _ => style::dim(status),
                };
                println!(
                    "{:<25} {:<40} {:<8} {:<10} {:<14} {}",
                    m["alias"].as_str().unwrap_or("-"),
                    m["repo_id"].as_str().unwrap_or("-"),
                    format!("{:.1}B", m["parameters_billion"].as_f64().unwrap_or(0.0)),
                    m["quantization"].as_str().unwrap_or("none"),
                    m["weight_format"].as_str().unwrap_or("-"),
                    status_colored,
                );
            }
            if models.is_empty() {
                println!("{}", style::dim("(no models)"));
            }
        }

        Commands::Ps => {
            let resp = client.get(format!("{}/api/ps", daemon_url)).send().await?;
            if !resp.status().is_success() {
                return Err(anyhow!("Ps failed: {}", resp.text().await.unwrap_or_default()));
            }
            let entries: Vec<serde_json::Value> = resp.json().await?;
            use crate::cli::style;
            println!(
                "{}",
                style::dim(&format!(
                    "{:<25} {:<10} {:<6} {:<35} {:<35}",
                    "ALIAS", "BACKEND", "PORT", "STARTED", "LAST_USED"
                ))
            );
            for e in &entries {
                let backend = e["backend"].as_str().unwrap_or("-");
                // Color BACKEND: sglang=cyan (GPU), llamacpp=green (CPU/portable).
                let backend_colored = match backend {
                    "sglang" => format!("{}{}{}", style::CYAN, backend, style::RESET),
                    "llamacpp" => style::ok(backend),
                    _ => backend.to_string(),
                };
                // BACKEND is a middle column, so pad to its visible width (10).
                let backend_cell = style::pad_colored(&backend_colored, backend.len(), 10);
                println!(
                    "{:<25} {} {:<6} {:<35} {:<35}",
                    e["alias"].as_str().unwrap_or("-"),
                    backend_cell,
                    e["port"].as_u64().unwrap_or(0),
                    e["started_at"].as_str().unwrap_or("-"),
                    e["last_used"].as_str().unwrap_or("-"),
                );
            }
            if entries.is_empty() {
                println!("{}", style::dim("(no running models)"));
            }
        }

        Commands::Load { model } => {
            let resp = client
                .post(format!("{}/api/load", daemon_url))
                .json(&LoadRequest { alias: model.clone() })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Load failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            let result: serde_json::Value = resp.json().await?;
            println!(
                "Model loaded. Endpoint: {}",
                result["endpoint"].as_str().unwrap_or("-")
            );
        }

        Commands::Rm { model, yes } => {
            if !yes {
                print!("Delete {}? [y/N]: ", model);
                io::stdout().flush()?;
                let mut stdin_lines =
                    tokio::io::BufReader::new(tokio::io::stdin()).lines();
                let answer = stdin_lines.next_line().await?.unwrap_or_default();
                if answer.trim().to_lowercase() != "y" {
                    println!("Aborted.");
                    return Ok(());
                }
            }
            let resp = client
                .delete(format!("{}/api/models/{}", daemon_url, model))
                .send()
                .await?;
            if !resp.status().is_success() && resp.status().as_u16() != 204 {
                return Err(anyhow!(
                    "Delete failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            println!("Deleted model: {}", model);
        }

        // B8 — show
        Commands::Show { model } => {
            let resp = client
                .post(format!("{}/api/show", daemon_url))
                .json(&OllamaShowRequest {
                    name: model.clone(),
                    verbose: false,
                })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Show failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            let info: serde_json::Value = resp.json().await?;
            println!("Model: {}", model);
            println!("\nDetails:");
            if let Some(d) = info.get("details") {
                println!("  family:           {}", d["family"].as_str().unwrap_or("-"));
                println!("  format:           {}", d["format"].as_str().unwrap_or("-"));
                println!("  parameter_size:   {}", d["parameter_size"].as_str().unwrap_or("-"));
                println!("  quantization:     {}", d["quantization_level"].as_str().unwrap_or("-"));
            }
            if let Some(mi) = info.get("model_info") {
                println!("\nModel info:");
                println!("  context_length: {}", mi["context_length"].as_u64().unwrap_or(0));
                println!("  files:          {}", mi["files"].as_u64().unwrap_or(0));
                println!("  revision:       {}", mi["revision"].as_str().unwrap_or("-"));
            }
            let template = info["template"].as_str().unwrap_or("");
            if !template.is_empty() {
                println!("\nTemplate:\n{}", template);
            }
            let params = info["parameters"].as_str().unwrap_or("");
            if !params.is_empty() {
                println!("\nParameters:\n{}", params);
            }
            let modelfile = info["modelfile"].as_str().unwrap_or("");
            if !modelfile.is_empty() {
                println!("\nModelfile:\n{}", modelfile);
            }
        }

        // B9 — cp
        Commands::Cp { source, destination } => {
            let resp = client
                .post(format!("{}/api/copy", daemon_url))
                .json(&OllamaCopyRequest {
                    source: source.clone(),
                    destination: destination.clone(),
                })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Copy failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            println!("Copied {} → {}", source, destination);
        }

        // B10 — create
        Commands::Create { name, file } => {
            let modelfile_text = tokio::fs::read_to_string(&file)
                .await
                .map_err(|e| anyhow!("Can't read Modelfile {}: {}", file, e))?;
            let resp = client
                .post(format!("{}/api/create", daemon_url))
                .json(&OllamaCreateRequest {
                    name: name.clone(),
                    modelfile: modelfile_text,
                    stream: false,
                })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Create failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            println!("Created model: {}", name);
        }

        Commands::Logs { model, lines } => {
            let resp = client
                .get(format!("{}/api/logs/{}?lines={}", daemon_url, model, lines))
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Logs failed ({}): {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            let body: serde_json::Value = resp.json().await?;
            let backend = body["backend"].as_str().unwrap_or("?");
            println!("# {} (backend: {})", model, backend);
            if let Some(arr) = body["lines"].as_array() {
                for line in arr {
                    if let Some(s) = line.as_str() {
                        println!("{}", s);
                    }
                }
            }
        }

        Commands::Chat { target, quantize } => {
            cmd_chat(&client, &daemon_url, &target, quantize.as_deref(), hf_token.as_deref()).await?;
        }

        Commands::Doctor => {
            cmd_doctor(&client, &daemon_url).await?;
        }

        Commands::Completion { shell } => {
            let mut cmd = <Cli as clap::CommandFactory>::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(shell, &mut cmd, name, &mut io::stdout());
        }

        Commands::Tail { lines } => {
            cmd_tail(&client, &daemon_url, lines).await?;
        }

        Commands::Setup { rebuild } => {
            println!("Setting up the inference engine (llama.cpp)...");
            let dir = if rebuild {
                crate::setup::ensure_llama_cpp_rebuild(&_settings).await?
            } else {
                crate::setup::ensure_llama_cpp(&_settings).await?
            };
            println!("Ready. llama.cpp is at {:?}", dir);
        }

        Commands::Install => {
            cmd_install()?;
        }
    }

    Ok(())
}

/// Copy the running binary onto the user's PATH so `localllm` works from any
/// directory. Cross-platform: `~/.local/bin` on Unix, `%LOCALAPPDATA%\localllm\bin`
/// on Windows. Prints a one-line hint if that dir isn't already on PATH.
fn cmd_install() -> Result<()> {
    let exe = std::env::current_exe()?;
    let exe_name = crate::platform::exe_name("localllm");

    let target_dir = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default())
            .join("localllm")
            .join("bin")
    } else {
        dirs::home_dir()
            .unwrap_or_default()
            .join(".local")
            .join("bin")
    };
    std::fs::create_dir_all(&target_dir)?;
    let dest = target_dir.join(&exe_name);

    std::fs::copy(&exe, &dest)
        .map_err(|e| anyhow!("Failed to copy binary to {:?}: {}", dest, e))?;

    println!("Installed localllm to {:?}", dest);

    // Tell the user if the dir isn't on PATH yet.
    let on_path = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d == target_dir))
        .unwrap_or(false);
    if !on_path {
        if cfg!(windows) {
            println!(
                "\nAdd it to PATH (PowerShell, one time):\n  \
                 setx PATH \"$env:PATH;{}\"",
                target_dir.display()
            );
        } else {
            println!(
                "\nAdd it to PATH (one time):\n  \
                 echo 'export PATH=\"$PATH:{}\"' >> ~/.bashrc && source ~/.bashrc",
                target_dir.display()
            );
        }
    } else {
        println!("You can now run `localllm` from anywhere.");
    }
    Ok(())
}

/// One-shot chat: send a single user message and stream the assistant reply to stdout.
/// On completion, prints token-usage stats to stderr (so stdout stays clean for piping).
async fn run_single(
    client: &reqwest::Client,
    daemon_url: &str,
    model: &str,
    prompt: &str,
) -> Result<()> {
    let req = ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ],
        stream: Some(true),
        temperature: Some(CHAT_TEMPERATURE),
        top_p: Some(CHAT_TOP_P),
        max_tokens: Some(CHAT_MAX_TOKENS),
        stop: None,
    };
    // Show a spinner while the request is in flight — the first request to a
    // model can take seconds while its backend cold-spawns. TTY-only so piped
    // output stays clean. Aborted the moment we have a response.
    let spinner = if stdout_is_tty() {
        Some(crate::cli::style::start_spinner("thinking…"))
    } else {
        None
    };
    let send_result = client
        .post(format!("{}/v1/chat/completions", daemon_url))
        .header("Accept", "text/event-stream")
        .json(&req)
        .send()
        .await;
    if let Some(s) = &spinner {
        s.abort();
        crate::cli::style::clear_line();
    }
    let resp = send_result?;
    if !resp.status().is_success() {
        return Err(anyhow!(
            "Chat completion failed ({}): {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }
    let mut content = String::new();
    let mut usage = None;
    // Render markdown/LaTeX only when writing to a real terminal; keep raw text
    // when piped/redirected so scripts get clean, unstyled output.
    stream_sse(resp, &mut content, &mut usage, stdout_is_tty()).await?;
    println!();
    if let Some(u) = usage {
        if let (Some(p), Some(c), Some(t)) = (
            u.get("prompt_tokens").and_then(|v| v.as_u64()),
            u.get("completion_tokens").and_then(|v| v.as_u64()),
            u.get("total_tokens").and_then(|v| v.as_u64()),
        ) {
            eprintln!("[tokens: prompt={} completion={} total={}]", p, c, t);
        }
    }
    Ok(())
}

/// Multi-turn REPL chat. Maintains the full conversation history in memory and
/// resends it on every turn (cheap — the daemon handles caching upstream).
///
/// Exits on empty line, EOF (Ctrl+D), or Ctrl+C. If a turn fails, the user's
/// message is popped off so the next turn isn't poisoned by a bad request.
async fn run_interactive(
    client: &reqwest::Client,
    daemon_url: &str,
    model: &str,
) -> Result<()> {
    use crate::cli::style::{BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW};

    // Seed the conversation with a system prompt so the assistant behaves like a
    // capable general assistant (concise, well-formatted Markdown, proper LaTeX
    // for math) — the same kind of priming Claude/GPT chat UIs apply.
    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    // Banner.
    println!("{CYAN}{BOLD}╭──────────────────────────────────────────────╮{RESET}");
    println!("{CYAN}{BOLD}│  localllm chat{RESET}");
    println!("{CYAN}{BOLD}│{RESET}  model: {GREEN}{model}{RESET}");
    println!(
        "{CYAN}{BOLD}│{RESET}  {DIM}/exit quit · /clear reset history · Ctrl+C to quit{RESET}"
    );
    println!("{CYAN}{BOLD}╰──────────────────────────────────────────────╯{RESET}");

    let mut stdin_lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!("\n{GREEN}{BOLD}❯{RESET} ");
        io::stdout().flush()?;

        // Read a line, but bail cleanly on Ctrl+C while waiting for input.
        let line = tokio::select! {
            res = stdin_lines.next_line() => match res? {
                None => break, // EOF (Ctrl+D)
                Some(l) => l.trim().to_string(),
            },
            _ = tokio::signal::ctrl_c() => {
                println!("\n{DIM}Goodbye.{RESET}");
                break;
            }
        };

        // Slash commands.
        match line.as_str() {
            "" => continue, // empty line: just re-prompt (don't exit)
            "/exit" | "/quit" | "/q" => {
                println!("{DIM}Goodbye.{RESET}");
                break;
            }
            "/clear" | "/reset" => {
                history.truncate(1); // keep the system prompt, drop the chat
                println!("{YELLOW}(history cleared){RESET}");
                continue;
            }
            "/help" => {
                println!("{DIM}commands: /exit  /clear  /help{RESET}");
                continue;
            }
            _ => {}
        }

        history.push(ChatMessage {
            role: "user".to_string(),
            content: line,
        });

        let req = ChatCompletionRequest {
            model: model.to_string(),
            messages: history.clone(),
            stream: Some(true),
            // Sampling tuned for natural, coherent chat (close to what hosted
            // assistants use): a little randomness, nucleus sampling to cut the
            // long tail. Generous token ceiling so answers aren't truncated.
            temperature: Some(CHAT_TEMPERATURE),
            top_p: Some(CHAT_TOP_P),
            max_tokens: Some(CHAT_MAX_TOKENS),
            stop: None,
        };

        // Show a "thinking" spinner until the first token arrives or the request
        // fails. Cancelled the moment streaming starts so it never overlaps text.
        let spinner = crate::cli::style::start_spinner("thinking…");

        let send_result = client
            .post(format!("{}/v1/chat/completions", daemon_url))
            .header("Accept", "text/event-stream")
            .json(&req)
            .send()
            .await;

        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                spinner.abort();
                crate::cli::style::clear_line();
                eprintln!("{RED}Error: {}{RESET}", e);
                history.pop();
                continue;
            }
        };

        if !resp.status().is_success() {
            spinner.abort();
            crate::cli::style::clear_line();
            let status = resp.status();
            let msg = resp.text().await.unwrap_or_default();
            eprintln!("{RED}Error ({}): {}{RESET}", status, msg);
            history.pop();
            continue;
        }

        // Stop the spinner and clear its line before the assistant text streams.
        spinner.abort();
        crate::cli::style::clear_line();
        print!("{CYAN}{BOLD}assistant{RESET} ");
        io::stdout().flush()?;

        let mut assistant_content = String::new();
        let mut _usage = None;
        stream_sse(resp, &mut assistant_content, &mut _usage, true).await?;
        println!();
        history.push(ChatMessage {
            role: "assistant".to_string(),
            content: assistant_content,
        });
    }
    Ok(())
}

/// Parse OpenAI-style Server-Sent Events (SSE) and print delta tokens as they arrive.
///
/// Wire format: each event is `data: {json}\n\n`, with `data: [DONE]` as the
/// terminator. We accumulate bytes in a buffer, use `memchr` to find newlines
/// (faster than `str::lines()` for byte streams), then drain each line in place.
///
/// Side effects:
///   * Writes each delta's `content` to stdout — through a Markdown+LaTeX
///     renderer when `render` is true (interactive chat), or raw when false
///     (one-shot `run`, so piped output stays clean for scripting).
///   * Appends the full text to `full_content` so the caller can save it.
///   * Stores the final `usage` object (token counts) in `usage_out` if present.
async fn stream_sse(
    resp: reqwest::Response,
    full_content: &mut String,
    usage_out: &mut Option<serde_json::Value>,
    render: bool,
) -> Result<()> {
    let mut byte_stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut md = render.then(crate::cli::render::MarkdownStream::new);

    while let Some(chunk) = byte_stream.next().await {
        let bytes = chunk?;
        buf.extend_from_slice(&bytes);

        // Process every complete line in the buffer using memchr for fast newline scan.
        while let Some(pos) = memchr::memchr(b'\n', &buf) {
            // Take ownership of bytes through end-of-line, drop them from buf.
            let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
            // Trim trailing \r\n
            let trim_end = line_bytes
                .iter()
                .rposition(|&b| b != b'\n' && b != b'\r')
                .map(|i| i + 1)
                .unwrap_or(0);
            let line = match std::str::from_utf8(&line_bytes[..trim_end]) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if line.is_empty() || line == "data: [DONE]" {
                continue;
            }
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            match serde_json::from_str::<ChatCompletionResponse>(data) {
                Ok(parsed) => {
                    if let Some(choice) = parsed.choices.first() {
                        if let Some(delta) = &choice.delta {
                            if !delta.content.is_empty() {
                                match md.as_mut() {
                                    Some(m) => m.push(&delta.content),
                                    None => {
                                        print!("{}", delta.content);
                                        io::stdout().flush()?;
                                    }
                                }
                                full_content.push_str(&delta.content);
                            }
                        }
                    }
                    if let Some(usage) = parsed.usage {
                        *usage_out = Some(serde_json::json!({
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }));
                    }
                }
                Err(_) => {
                    tracing::debug!("Non-JSON SSE line: {}", data);
                }
            }
        }
    }
    // Flush any buffered final line (text with no trailing newline).
    if let Some(m) = md.as_mut() {
        m.finish();
    }
    Ok(())
}

/// POST `/api/pull` with `stream: true` and print each NDJSON status frame.
/// Shows real-time progress (downloading / quantizing) instead of silently
/// waiting for the daemon to finish and return a single JSON blob.
async fn stream_pull(
    client: &reqwest::Client,
    daemon_url: &str,
    repo_id: &str,
    revision: Option<&str>,
    quantize: Option<&str>,
    hf_token: Option<&str>,
) -> Result<()> {
    let body = serde_json::json!({
        "repo_id": repo_id,
        "revision": revision,
        "quantize": quantize,
        "stream": true,
        "hf_token": hf_token,
    });

    let resp = client
        .post(format!("{}/api/pull", daemon_url))
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(anyhow!(
            "Pull failed ({}): {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }

    let mut byte_stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut last_status = String::new();
    // Download-rate tracking for the progress bar.
    let dl_started = std::time::Instant::now();
    let mut bar_active = false;
    // Printed once when total size first becomes known.
    let mut total_announced = false;

    while let Some(chunk) = byte_stream.next().await {
        buf.extend_from_slice(&chunk?);
        while let Some(pos) = memchr::memchr(b'\n', &buf) {
            let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
            let line = std::str::from_utf8(&line_bytes).unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let Ok(frame) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let status = frame["status"].as_str().unwrap_or("").to_string();
            match status.as_str() {
                "error" => {
                    if bar_active {
                        println!();
                    }
                    return Err(anyhow!(
                        "Pull failed: {}",
                        frame["error"].as_str().unwrap_or("unknown error")
                    ));
                }
                "download" => {
                    let downloaded = frame["downloaded"].as_u64().unwrap_or(0);
                    let total = frame["total"].as_u64().unwrap_or(0);
                    // Print the total size once, before the first bar update.
                    if !total_announced && total > 0 {
                        println!("  Total download: {}", fmt_bytes(total));
                        total_announced = true;
                    }
                    let elapsed = dl_started.elapsed().as_secs_f64().max(0.001);
                    let speed = downloaded as f64 / elapsed; // bytes/sec
                    print!("\r\x1b[K  {}", render_progress_bar(downloaded, total, speed));
                    io::stdout().flush()?;
                    bar_active = true;
                }
                "success" => {
                    if bar_active {
                        println!(); // finish the progress-bar line
                    }
                    println!("\nModel pulled successfully:");
                    println!("  alias:        {}", frame["alias"].as_str().unwrap_or("-"));
                    println!("  architecture: {}", frame["architecture"].as_str().unwrap_or("-"));
                    println!(
                        "  parameters:   {:.1}B",
                        frame["parameters_billion"].as_f64().unwrap_or(0.0)
                    );
                    println!("  context:      {}", frame["context_length"].as_u64().unwrap_or(0));
                    if let Some(q) = frame["quantization"].as_str() {
                        println!("  quantization: {}", q);
                    }
                }
                s if s != last_status => {
                    // New phase label — print on its own line (above any future bar).
                    if bar_active {
                        println!();
                        bar_active = false;
                    }
                    println!("  {s}...");
                    last_status = s.to_string();
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// True when stdout is an interactive terminal (not a pipe/file). Uses the
/// std `IsTerminal` trait (stable since Rust 1.70) — no extra dependency, works
/// on Windows + Unix. Drives whether we apply ANSI markdown/LaTeX styling.
fn stdout_is_tty() -> bool {
    use std::io::IsTerminal;
    io::stdout().is_terminal()
}

/// Render a single-line download progress bar:
///   `[==========>          ]  45%  210.5 MB / 470.1 MB  12.3 MB/s`
/// Falls back to a byte/speed readout when total size is unknown (total == 0).
fn render_progress_bar(downloaded: u64, total: u64, speed_bps: f64) -> String {
    let width = 28usize;
    if total == 0 {
        return format!("{}  {}/s", fmt_bytes(downloaded), fmt_bytes(speed_bps as u64));
    }
    let frac = (downloaded as f64 / total as f64).clamp(0.0, 1.0);
    let filled = (frac * width as f64) as usize;
    let mut bar = String::with_capacity(width);
    for i in 0..width {
        if i < filled {
            bar.push('=');
        } else if i == filled {
            bar.push('>');
        } else {
            bar.push(' ');
        }
    }
    format!(
        "[{}] {:>3.0}%  {} / {}  {}/s",
        bar,
        frac * 100.0,
        fmt_bytes(downloaded),
        fmt_bytes(total),
        fmt_bytes(speed_bps as u64),
    )
}

/// Human-readable byte size: B / KB / MB / GB with one decimal.
fn fmt_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let f = n as f64;
    if f >= GB {
        format!("{:.1} GB", f / GB)
    } else if f >= MB {
        format!("{:.1} MB", f / MB)
    } else if f >= KB {
        format!("{:.1} KB", f / KB)
    } else {
        format!("{} B", n)
    }
}

/// **B1** — `localllm chat <target>`. The shortest path from "I want to chat"
/// to a running model:
///   1. If `target` looks like a HF repo (`org/Name`), check if already pulled.
///      If not, pull (auto-quantize when no GPU is detected).
///   2. Drop straight into the interactive REPL.
async fn cmd_chat(
    client: &reqwest::Client,
    daemon_url: &str,
    target: &str,
    quantize_override: Option<&str>,
    hf_token: Option<&str>,
) -> Result<()> {
    let is_repo = target.contains('/');
    let alias = if is_repo {
        // Derive what the alias *would* be once pulled.
        let last = target.split('/').next_back().unwrap_or(target);
        last.to_lowercase().replace(['.', ' '], "-")
    } else {
        target.to_string()
    };

    // Check if already pulled.
    let resp = client
        .get(format!("{}/api/models", daemon_url))
        .send()
        .await?;
    let pulled: bool = if resp.status().is_success() {
        let models: Vec<serde_json::Value> = resp.json().await.unwrap_or_default();
        models
            .iter()
            .any(|m| m["alias"].as_str() == Some(&alias))
    } else {
        false
    };

    if !pulled && is_repo {
        // Decide whether to auto-quantize. Check the daemon's /health which
        // reports GPU presence indirectly (we'll use the doctor's GPU probe
        // by checking nvidia-smi locally).
        let need_quant = match quantize_override {
            Some(_) => true,
            None => !has_local_gpu(),
        };
        let quant_level = quantize_override
            .map(|s| s.to_string())
            .or(if need_quant {
                Some("Q4_K_M".to_string())
            } else {
                None
            });

        println!("Pulling {}...", target);
        if let Some(q) = &quant_level {
            println!("(no GPU detected — will quantize to {})", q);
        }
        stream_pull(client, daemon_url, target, None, quant_level.as_deref(), hf_token).await?;
        println!("Model ready.");
    } else if !pulled && !is_repo {
        return Err(anyhow!(
            "'{}' is not a HuggingFace repo path and isn't pulled locally. \
             Try `localllm chat org/ModelName` instead.",
            target
        ));
    }

    // Hand off to interactive REPL with the resolved alias.
    run_interactive(client, daemon_url, &alias).await
}

/// Cheap local GPU detection used by `chat` to decide whether to auto-quantize.
/// Returns true if `nvidia-smi` exists AND reports at least one GPU.
fn has_local_gpu() -> bool {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader"])
        .output();
    matches!(output, Ok(o) if o.status.success() && !o.stdout.is_empty())
}

/// Sum free VRAM (MiB) across all GPUs via nvidia-smi. `None` if unavailable.
fn gpu_free_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let total: u64 = text
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .sum();
    if total > 0 {
        Some(total)
    } else {
        None
    }
}

/// **B2** — `localllm doctor`. Walk through preconditions and report what's
/// configured vs missing. Designed to answer "why isn't this working".
async fn cmd_doctor(client: &reqwest::Client, daemon_url: &str) -> Result<()> {
    use crate::cli::style;
    let mut ok = 0;
    let mut warn = 0;

    println!("{}\n", style::dim("Running localllm doctor..."));

    // 1. Daemon reachable
    print!("  {} Daemon at {} ... ", style::dim("[1/7]"), daemon_url);
    io::stdout().flush()?;
    match client.get(format!("{}/health", daemon_url)).send().await {
        Ok(r) if r.status().is_success() => {
            println!("{}", style::ok("OK"));
            ok += 1;
        }
        Ok(r) => {
            println!("{} (HTTP {})", style::warn("WARN"), r.status());
            warn += 1;
        }
        Err(e) => {
            println!("{} ({})", style::err("MISSING"), e);
            warn += 1;
        }
    }

    // 2. nvidia-smi (GPU)
    print!("  {} NVIDIA GPU      ... ", style::dim("[2/7]"));
    io::stdout().flush()?;
    if has_local_gpu() {
        // Report total free VRAM so the user can gauge what'll fit / offload.
        match gpu_free_vram_mb() {
            Some(free) => {
                println!(
                    "{} ({} MiB free — models will GPU-offload as many layers as fit)",
                    style::ok("OK"),
                    free
                );
            }
            None => println!("{} (GPU offload enabled)", style::ok("OK")),
        }
        ok += 1;
    } else {
        println!("{}", style::dim("absent (will run on CPU via llama.cpp)"));
    }

    // Inference tuning summary (informational; always shown).
    println!(
        "{}",
        style::dim(&format!(
            "        CPU decode threads (auto): {} physical cores",
            crate::config::physical_cores()
        ))
    );

    // 3. Python + sglang. Resolve the interpreter name per-OS.
    let python = crate::platform::python_command();
    print!("  {} python ({:<7})... ", style::dim("[3/7]"), python);
    io::stdout().flush()?;
    let py_status = std::process::Command::new(&python)
        .arg("--version")
        .output();
    match py_status {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout);
            let v = v.trim();
            println!("{} ({})", style::ok("OK"), v);
            ok += 1;

            // 4. sglang
            print!("  {} sglang          ... ", style::dim("[4/7]"));
            io::stdout().flush()?;
            let sg = std::process::Command::new(&python)
                .args(["-c", "import sglang; print(sglang.__version__)"])
                .output();
            match sg {
                Ok(o) if o.status.success() => {
                    let v = String::from_utf8_lossy(&o.stdout);
                    println!("{} ({})", style::ok("OK"), v.trim());
                    ok += 1;
                }
                _ => {
                    println!("{} — install with: pip install sglang", style::dim("missing"));
                    warn += 1;
                }
            }
        }
        _ => {
            println!("{} — required for sglang backend", style::err("MISSING"));
            warn += 1;
            println!("  {} sglang          ... {}", style::dim("[4/7]"), style::dim("skipped (no Python)"));
        }
    }

    // 5. llama.cpp
    print!("  {} llama.cpp       ... ", style::dim("[5/7]"));
    io::stdout().flush()?;
    let cfg = crate::config::Settings::load()?;
    match cfg.llama_cpp_dir.as_ref() {
        Some(dir) if dir.join("build").join("bin").exists() => {
            println!("{} ({:?})", style::ok("OK"), dir);
            ok += 1;
        }
        Some(dir) => {
            println!("{} build/bin/ missing at {:?}", style::warn("WARN"), dir);
            warn += 1;
        }
        None => {
            println!(
                "{} — runs automatically on first use, or run: localllm setup",
                style::warn("absent")
            );
            warn += 1;
        }
    }

    // 6. HF_TOKEN (optional)
    print!("  {} HF_TOKEN        ... ", style::dim("[6/7]"));
    io::stdout().flush()?;
    if std::env::var("HF_TOKEN").is_ok() {
        println!("{} (private models accessible)", style::ok("set"));
        ok += 1;
    } else {
        println!("{}", style::dim("not set (public models only)"));
    }

    // 7. Disk space available in models_dir
    print!("  {} Models dir      ... ", style::dim("[7/7]"));
    io::stdout().flush()?;
    let md = &cfg.models_dir;
    if md.exists() || std::fs::create_dir_all(md).is_ok() {
        println!("{} ({:?})", style::ok("OK"), md);
        ok += 1;
    } else {
        println!("{} — can't create {:?}", style::err("WARN"), md);
        warn += 1;
    }

    let summary = format!("\nSummary: {} OK, {} warning(s)", ok, warn);
    if warn == 0 {
        println!("{}", style::ok(&summary));
        println!("{}", style::ok("Everything looks good."));
    } else {
        println!("{}", style::warn(&summary));
        println!("{}", style::dim("Review warnings above. Fix each line marked WARN/MISSING."));
    }
    Ok(())
}

/// **B5** — `localllm tail`. Aggregates `/api/logs/<alias>` snapshots across
/// every running model, prints them interleaved. One-shot — not a streaming
/// follow. Good enough for "what just happened" debugging.
async fn cmd_tail(client: &reqwest::Client, daemon_url: &str, lines: usize) -> Result<()> {
    let resp = client.get(format!("{}/api/ps", daemon_url)).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow!(
            "ps failed ({}): {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }
    let processes: Vec<serde_json::Value> = resp.json().await?;
    if processes.is_empty() {
        println!("(no running models)");
        return Ok(());
    }

    for p in processes {
        let Some(alias) = p["alias"].as_str() else {
            continue;
        };
        let backend = p["backend"].as_str().unwrap_or("?");
        println!("\n=== {} ({}) ===", alias, backend);
        let logs_resp = client
            .get(format!("{}/api/logs/{}?lines={}", daemon_url, alias, lines))
            .send()
            .await?;
        if !logs_resp.status().is_success() {
            println!("(failed to fetch logs)");
            continue;
        }
        let body: serde_json::Value = logs_resp.json().await?;
        if let Some(arr) = body["lines"].as_array() {
            for line in arr {
                if let Some(s) = line.as_str() {
                    println!("{}", s);
                }
            }
        }
    }
    Ok(())
}
