# localllm

A production-grade local LLM serving tool in a **single Rust binary**. Pull models
from HuggingFace, run them on **GPU (VRAM) or CPU (RAM) automatically**, and chat
from the terminal — with an OpenAI- and Ollama-compatible HTTP API for your apps.

- **Zero config.** `cargo build`, then run. The inference engine (llama.cpp) is
  built automatically on first use.
- **GPU or CPU, automatic.** Uses your NVIDIA GPU's VRAM when present, otherwise
  falls back to CPU + system RAM. No flags to set.
- **No Python needed for CPU.** Models are pulled as prebuilt GGUF files directly
  from HuggingFace — no PyTorch, no conversion step.
- **Nice terminal chat.** Streamed replies are rendered with Markdown styling and
  LaTeX math turned into readable Unicode (`x^2` → x², `\alpha` → α, `\frac{a}{b}` → (a)/(b)).
- **Cross-platform.** Works the same on Ubuntu/Linux and Windows.

---

## Quick start

```bash
# 1. Build the binary (one command)
cargo build --release

# 2. (optional) put it on your PATH so you can type `localllm` anywhere
./target/release/localllm install

# 3. Chat. First run auto-builds llama.cpp (~3–8 min, one time), then pulls the
#    model and drops you into an interactive chat.
localllm chat Qwen/Qwen2.5-0.5B-Instruct
```

That's the whole setup. No `git clone llama.cpp`, no `cmake`, no editing config files.

> **Prerequisites for the one-time auto-build:** `git`, `cmake`, and a C++ compiler.
> - Ubuntu: `sudo apt-get install -y git cmake build-essential`
> - Windows: `winget install Git.Git Kitware.CMake` + Visual Studio Build Tools
> - macOS: `brew install git cmake`
>
> If you'd rather build the engine up front instead of on first chat:
> `localllm setup`

---

## Everyday commands

```bash
localllm chat <hf-repo>          # pull (if needed) + interactive chat — the easy path
localllm pull <hf-repo>          # download a model (shows a progress bar)
localllm run <model> "<prompt>"  # one-shot prompt, prints the answer
localllm run <model> -i          # interactive REPL for an already-pulled model
localllm list                    # list downloaded models
localllm ps                      # show running model processes
localllm logs <model>            # tail a model's backend logs
localllm rm <model>              # delete a model
localllm doctor                  # diagnose your setup (GPU, engine, paths, ...)
```

Inside an interactive chat: `/exit` to quit, `/clear` to reset the conversation,
`/help` for the command list. `Ctrl+C` also exits cleanly.

---

## How GPU vs CPU is chosen

localllm probes for an NVIDIA GPU (via `nvidia-smi`) and picks both the **file
format** to download and the **backend** to run accordingly:

- **GPU present** → a plain `pull` downloads full-precision **safetensors** and the
  higher-throughput **sglang** backend serves them (layers offloaded to VRAM; the
  rest spill to CPU RAM automatically).
- **No GPU** → a plain `pull` downloads a ready-made **GGUF** (default `Q4_K_M`) and
  **llama.cpp** runs it on **CPU using system RAM**. No Python, no PyTorch, no
  conversion step — exactly the path that works out of the box.

Why the difference: `llama-server` only runs GGUF files, and converting safetensors
→ GGUF needs Python + PyTorch. So on a CPU-only box localllm fetches a prebuilt GGUF
directly (from the model's repo or a community mirror like bartowski/unsloth). Raw
safetensors run only on GPU (sglang), or after a local convert if you have torch
installed (`localllm quantize <model>`).

You don't configure any of this — just `pull`/`run`/`chat` and it picks the right
path for your hardware.

---

## HTTP API

The daemon (auto-started on first command, listening on `http://127.0.0.1:11435`)
speaks both wire formats, so existing tooling works unchanged:

- **OpenAI:** `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `GET /v1/models`
- **Ollama:** `POST /api/chat`, `/api/generate`, `GET /api/tags`, `POST /api/show`, …

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2-5-0-5b-instruct","messages":[{"role":"user","content":"hi"}]}'
```

---

## Where things live

```
~/.localllm/
├── llama.cpp/     ← auto-built inference engine (first run)
├── gguf/          ← downloaded / quantized GGUF model files
├── models/        ← HuggingFace safetensors (GPU/sglang path only)
├── manifests/     ← one <alias>.json per registered model
├── config.toml    ← optional overrides (auto-managed; rarely needed)
└── daemon.pid     ← daemon lock file (while running)
```

Override locations with env vars (`LOCALLLM_MODELS_DIR`, `LOCALLLM_GGUF_DIR`, …)
or a `~/.localllm/config.toml`. Set `HF_TOKEN` to access gated/private models.

---

## Further reading

- [USAGE.md](USAGE.md) — full command reference and configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) — how it works internally
- [WORKFLOW_DEEP_DIVE.md](WORKFLOW_DEEP_DIVE.md) — end-to-end request lifecycle
