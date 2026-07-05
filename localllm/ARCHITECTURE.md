# localllm — Architecture & Workflow

A complete walkthrough of how every part of localllm works, end to end. Companion to [USAGE.md](USAGE.md) (which covers *how to use it*); this document covers *how it actually does what it does*.

---

## Table of contents

1. [The 10,000-foot view](#1-the-10000-foot-view)
2. [Single binary, two roles](#2-single-binary-two-roles)
3. [Startup flow — `localllm <anything>`](#3-startup-flow--localllm-anything)
4. [Daemon bootstrap — what happens inside `serve`](#4-daemon-bootstrap--what-happens-inside-serve)
5. [The model lifecycle](#5-the-model-lifecycle)
6. [Pulling a model — step by step](#6-pulling-a-model--step-by-step)
7. [Quantizing — step by step](#7-quantizing--step-by-step)
8. [Loading & inference routing](#8-loading--inference-routing)
9. [Chat completion — request to streamed reply](#9-chat-completion--request-to-streamed-reply)
10. [Modelfile pipeline](#10-modelfile-pipeline)
11. [Concurrency & locking model](#11-concurrency--locking-model)
12. [Background tasks](#12-background-tasks)
13. [Storage layout on disk](#13-storage-layout-on-disk)
14. [Shutdown sequence](#14-shutdown-sequence)
15. [Cross-platform behavior differences](#15-cross-platform-behavior-differences)

---

## 1. The 10,000-foot view

localllm is **one Rust binary** that does both jobs of a typical "Ollama-style" tool:

- A **CLI** for humans (`localllm pull`, `localllm run`, etc.).
- A **persistent HTTP daemon** that exposes OpenAI- and Ollama-compatible APIs.

The two roles share the same executable but communicate over HTTP — the CLI is a thin wrapper that talks to the daemon. The daemon is what owns all state: the model registry, running inference processes, the connection pool, metrics counters.

Inference itself is delegated to one of two **external backends**:

- **sglang** (Python + CUDA) — high-throughput GPU server.
- **llama.cpp** (`llama-server` binary) — CPU/quant GPU fallback.

localllm spawns these as child processes, health-checks them, proxies user requests to them, and reaps them when they go idle. localllm is **not** itself a transformer runtime — it's the orchestration layer above one.

```
┌──────────────┐  HTTP    ┌──────────────────┐  spawn  ┌─────────────────┐
│  localllm    │ ───────► │  localllm        │ ──────► │  sglang.py      │
│  CLI         │          │  daemon          │         │  (GPU)          │
└──────────────┘          │  (axum + tokio)  │         └─────────────────┘
                          │                  │  spawn  ┌─────────────────┐
                          │  /v1/* /api/*    │ ──────► │  llama-server   │
                          │                  │         │  (CPU / GGUF)   │
                          └──────────────────┘         └─────────────────┘
                                  │
                                  ▼
                          ┌──────────────────┐
                          │  ~/.localllm/    │
                          │   models/        │  (HF safetensors)
                          │   gguf/          │  (quantized GGUF)
                          │   manifests/     │  (one JSON per model)
                          │   daemon.pid     │  (lock file)
                          └──────────────────┘
```

---

## 2. Single binary, two roles

Everything lives in `target/release/localllm.exe` (or `localllm` on Unix). Which role it plays is decided by the first CLI argument:

| Invocation | Role | Behavior |
|---|---|---|
| `localllm serve` | **Daemon** | Binds the HTTP port, runs forever until Ctrl+C. |
| `localllm <anything else>` | **CLI** | Sends an HTTP request to the daemon, prints result, exits. |

This is dispatched in [`main.rs`](src/main.rs):

```rust
if matches!(cli.command, Commands::Serve) {
    return daemon::server::run(Arc::new(settings)).await;
}
// ...otherwise we're in CLI mode → make HTTP calls
```

The CLI binary is intentionally **stateless** — it owns no model files, no port allocations, no registry. The daemon is the single source of truth. This keeps things simple: you can run as many CLI invocations in parallel as you want and they'll all coordinate through the daemon.

---

## 3. Startup flow — `localllm <anything>`

When you type `localllm pull <model>`, here's everything that happens before the pull even starts:

```
1. Initialize tracing  (RUST_LOG=localllm=info by default)
2. Parse CLI args      (clap derives from Commands enum)
3. Load Settings       (~/.localllm/config.toml + env overrides)
4. Resolve daemon URL  (--daemon-url > LOCALLLM_DAEMON_URL > config)
5. Probe /health       (GET, 500ms timeout)
   │
   ├── Daemon alive?  ──→ skip to step 8
   │
   └── Daemon down?
       │
       6. Auto-spawn detached daemon
          │ Windows: CreateProcess with DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP
          │ Unix:    fork → setsid → exec (so SIGHUP on parent doesn't kill child)
          │
       7. Poll /health for up to 5s (100ms interval)
          (small daemons become reachable in ~200ms)
8. cli::commands::execute(cli, settings, daemon_url)
   │
   └── Match on subcommand, build HTTP request, send, format response
```

The **auto-spawn** is what makes `localllm` feel like a single tool. There's no separate `start-daemon` step — the first CLI invocation that finds no daemon brings one up automatically, and it survives the CLI exiting. From the user's perspective: just run any command, and it works.

Key file: [`main.rs`](src/main.rs)

---

## 4. Daemon bootstrap — what happens inside `serve`

When `localllm serve` runs (either directly or via auto-spawn), the bootstrap order in [`daemon/server.rs`](src/daemon/server.rs) is:

```
1. settings.ensure_dirs()
   Creates models_dir, gguf_dir, manifests_dir if missing.

2. PidLock::acquire(&settings)
   Reads ~/.localllm/daemon.pid (if any).
   If file exists and PID is alive  → ABORT (clean error message).
   If file exists but PID is dead   → log "stale" warning, take ownership.
   If file missing                  → create it with our PID.
   Drop impl removes the file on shutdown.

3. ManifestStore::new(manifests_dir) → empty in-memory cache.

4. spawn_blocking(|| registry.load_all())
   Walks manifests_dir, parses each *.json into Arc<ModelManifest>.
   Runs in the background so the HTTP server starts serving immediately
   even with thousands of manifests.

5. HuggingFaceClient::new(token)
   Builds TWO reqwest clients: one with a 60s timeout for metadata calls,
   one with NO total timeout for file downloads (multi-GB streams).

6. DownloadManager::new(hf_client, settings)
   Just glue — no I/O.

7. CompressionEngine::new(llama_cpp_dir, settings)
   Created only if llama_cpp_dir is configured.

8. SglangManager::new(settings)        ──┐
9. LlamaCppManager::new(settings)      ──┤  Each holds an empty DashMap
                                         │  keyed by alias.
10. InferenceRouter::new(sg, lc, set)  ──┘

11. reqwest::Client for upstream proxying
    Tuned: 90s idle, 32 max idle/host, tcp_keepalive=60s, tcp_nodelay=true.

12. Metrics::default() → atomic counters.

13. Bundle everything into AppState; wrap in Arc.

14. build_router(state) → axum::Router with every endpoint mounted.

15. tokio::TcpListener::bind(daemon_host:daemon_port).

16. Spawn background TTL eviction task (60s interval).

17. axum::serve(listener, router).with_graceful_shutdown(...).await
    Blocks here until Ctrl+C / SIGTERM.

18. On shutdown:
    a. inference_router.kill_all().await — kills every sglang and llama-server
       process tree.
    b. drop(pid_lock) — removes daemon.pid.
```

The PID lock is critical: without it, two `localllm serve` invocations on the same machine would silently race for the port and produce confusing errors. With it, the second one gets a clean "Another localllm daemon is already running (pid N)" message.

Key files: [`daemon/server.rs`](src/daemon/server.rs), [`api/routes.rs`](src/api/routes.rs)

---

## 5. The model lifecycle

A model passes through these states:

```
   (nonexistent)
        │
        │ localllm pull <repo>
        ▼
   [ DOWNLOADED ]          safetensors in models_dir, manifest saved
        │
        │ localllm quantize <model> --level Q4_K_M
        ▼
   [ QUANTIZED ]           GGUF added to gguf_dir, manifest updated
        │
        │ first inference request OR localllm load
        ▼
   [ LOADED ]              sglang or llama-server process running, port allocated
        │
        │ no requests for model_ttl_secs (default 5 min)
        ▼
   [ DOWNLOADED ]          process killed by eviction loop, files retained
        │
        │ localllm rm <model> --yes
        ▼
   (nonexistent)           manifest deleted, files deleted, process killed
```

A few states **don't** require human intervention:

- **DOWNLOADED → LOADED**: happens automatically on the first chat/generate request.
- **LOADED → DOWNLOADED**: happens automatically when the model idle-times out.

This means the typical user lifecycle is just `pull` once, `run` many times, optionally `quantize` if you need a smaller VRAM footprint, and eventually `rm` when you don't need the model anymore.

---

## 6. Pulling a model — step by step

`localllm pull meta-llama/Llama-3.2-1B-Instruct` is the most complex command. Here's what happens:

**Format selection up front.** Before downloading anything, the daemon decides what
to fetch (`effective_pull_quant` in [`api/routes.rs`](src/api/routes.rs)): an explicit
`--quantize` always wins; otherwise on a **GPU** box it pulls full-precision
safetensors (for sglang), and on a **CPU-only** box it pulls a ready-made `Q4_K_M`
**GGUF** so the result actually runs. The GGUF path (`pull_gguf`) needs no
Python/torch; it only falls back to a local safetensors→GGUF convert when no prebuilt
GGUF exists anywhere AND `llama_cpp_dir` + torch are available.

### Phase A — CLI side ([`cli/commands.rs`](src/cli/commands.rs))

```
1. Build PullRequest { repo_id, revision, quantize }.
2. POST {daemon_url}/api/pull with JSON body.
3. Wait for response (no timeout — pulls can be very long).
4. Parse manifest from response, pretty-print alias/architecture/params/quant.
```

### Phase B — Daemon side ([`api/routes.rs::pull_model`](src/api/routes.rs))

```
5. Increment model_pulls_total counter.
6. Detect body shape — native CLI vs Ollama-style. Dispatch.
7. Call download_manager.pull(repo_id, revision).
```

### Phase C — Download orchestration ([`downloader/file_manager.rs::pull`](src/downloader/file_manager.rs))

```
8.  GET https://huggingface.co/api/models/<repo_id>
    │ Extract revision from X-Repo-Commit header.
    │ Extract siblings[] from JSON body.
9.  Build local destination path:
        ~/.localllm/models/<sanitized_repo_id>/<revision>/
        tokio::fs::create_dir_all(local_dir).
10. Filter files: allow-list (config.json, tokenizer*, *.safetensors)
                  deny-list  (*.msgpack, *.h5, *.ot, *.npz, *.pt)
                  is_safe_filename guard (no path traversal).
11. Canonicalize local_dir for the per-file containment check.
12. tokio::sync::Semaphore::new(max_concurrent_downloads) — caps parallelism.
13. For each file → tokio::spawn:
       a. Acquire semaphore permit.
       b. Verify dest path (after symlink resolution) stays inside local_dir.
       c. Call hf_client.download_file(...).
       d. Mark progress bar done.
14. join_all(handles) — fail fast on first error.
15. Read config.json from downloaded files.
16. Extract architecture (LlamaForCausalLM → "llama" etc).
17. Extract context_length from max_position_embeddings.
18. Estimate parameters from hidden_size, num_hidden_layers, vocab_size,
    intermediate_size using the standard transformer geometry formula.
19. Build ModelManifest and return it.
```

### Phase D — Per-file download ([`downloader/hf_api.rs::download_file`](src/downloader/hf_api.rs))

```
20. Cached fast path:
        if dest exists at expected_size → set progress to 100%, return Ok.
21. Retry loop (3 attempts, exponential backoff 2s/4s/8s):
       a. Call attempt_download():
          - If file partially exists → HTTP Range: bytes=<existing>- (resume).
          - Else → fresh GET.
          - Stream chunks: write to file, hash if fresh, update progress bar.
          - On success: verify SHA-256 if expected hash provided.
       b. On Err: delete partial file, sleep, retry.
22. Return Ok on first success or Err if all 3 attempts fail.
```

### Phase E — Save & respond

```
23. (Back in routes.rs) If req.quantize is set, call compression engine →
    update manifest.gguf_path and manifest.quantization.
24. registry.save(&manifest):
       a. Write to <alias>.<uuid>.tmp.
       b. fs::rename(tmp, <alias>.json).
       c. Insert into cache HashMap.
25. Return JSON(manifest) → CLI pretty-prints it.
```

The key robustness features:

- **Cached fast path** — repeated pulls are instant.
- **Range-based resume** — interrupted downloads pick up where they left off.
- **Per-file SHA-256** — corruption detection on fresh downloads.
- **Atomic save** — no half-written manifest files on a crash.
- **Path containment** — even if HF returns a malicious filename, we won't write outside `models_dir`.

---

## 7. Quantizing — step by step

`localllm quantize my-model --level Q4_K_M` invokes [`compression/quantize.rs`](src/compression/quantize.rs):

```
1. registry.get(alias) → load the manifest.
2. QuantizationLevel::from_str("Q4_K_M") → enum value.
3. Verify llama_cpp_dir is configured (else error with hint).
4. Step A — Convert to F16 GGUF:
       python3 <llama_cpp_dir>/convert_hf_to_gguf.py \
           <manifest.local_path> \
           --outtype f16 \
           --outfile <gguf_dir>/<alias>-F16.gguf
   spawn_and_log captures stdout/stderr; non-zero exit returns Err with stderr.
5. Step B — Quantize:
       <llama_cpp_dir>/build/bin/llama-quantize \
           <gguf_dir>/<alias>-F16.gguf \
           <gguf_dir>/<alias>-Q4_K_M.gguf \
           Q4_K_M
6. Step C — Delete the intermediate F16 file (typically 2x size of final).
7. Update manifest: gguf_path = <new path>, quantization = Q4KM.
8. registry.save(&manifest).
```

The whole sequence is async-await but each external command (`python3`, `llama-quantize`) is a blocking process. We use `tokio::process::Command` so the main runtime keeps serving other requests while the convert runs.

**Why two stages?** llama-quantize works on GGUF inputs, not safetensors. The convert script is the only way to get from HF tensors to GGUF. We could do F16 quantization in one step, but for lower precisions (Q4, Q5, Q6) the convert-then-quantize pipeline is standard.

---

## 8. Loading & inference routing

When a chat/generate request arrives for a model that isn't loaded yet, [`inference/mod.rs::get_endpoint`](src/inference/mod.rs) decides which backend to spawn:

```
get_endpoint(manifest):
    │
    1. gpus = VramManager::query_gpus()
       (shells out to nvidia-smi; empty Vec if missing/failed)
    │
    2. If !gpus.is_empty() AND VramManager::can_fit(manifest, &gpus):
           → sglang_manager.get_or_spawn(alias, manifest)
           → return http://127.0.0.1:<sglang_port>
    │
    3. Else if manifest.gguf_path.is_some():
           → llamacpp_manager.get_or_spawn(alias, manifest)
           → return http://127.0.0.1:<llamacpp_port>
    │
    4. Else:
           → Err("No runnable backend for <alias>: no GGUF and doesn't fit GPU VRAM.
                  Re-pull to fetch a prebuilt GGUF, or quantize with Python+torch.")
```

In practice this last error is rare: a plain `pull` on a CPU-only box already
downloads a runnable GGUF by default (see section 6), so `gguf_path` is set and
step 3 succeeds. It only triggers for a safetensors-only model that no longer fits
the GPU it was pulled for.

### `can_fit` heuristic ([`gpu/vram.rs`](src/gpu/vram.rs))

```
estimate_model_vram_mb(manifest):
    bytes_per_param = manifest.quantization.bytes_per_param() OR 2.0 (bfloat16)
    weight_mb       = params × 1e9 × bytes_per_param / 1024² 
    kv_cache_mb     = context_length × 64 / 1024
    return weight_mb + kv_cache_mb

can_fit(manifest, gpus):
    total_free = sum(gpu.free_vram_mb for gpu in gpus)
    required   = estimate_model_vram_mb(manifest) × 1.15  // 15% headroom
    return total_free >= required
```

This is an estimate — KV cache size depends on head count and batch size which we don't track precisely. If we predict wrong, the downstream sglang spawn will OOM and surface a clean error. The estimate just keeps us from trying obvious losers.

### `get_or_spawn` ([`inference/sglang.rs`](src/inference/sglang.rs))

```
get_or_spawn(alias, manifest):
    │
    1. processes.get_mut(alias):
       │  
       ├── None → continue to spawn
       │
       └── Some(entry):
           try_wait():
             Ok(None)         → alive: touch last_used (AtomicI64 store), return port
             Ok(Some(status)) → crashed: log, remove, continue to spawn
             Err(_)           → unknown: log, remove, continue to spawn
    │
    2. spawn(manifest):
           a. Acquire port_alloc_lock (Mutex).
           b. find_free_port(30000..=31000) → first port that binds.
           c. Drop the lock.
           d. Build python -m sglang.launch_server command:
                --model-path <local_path>
                --host 127.0.0.1 --port <port>
                --dtype bfloat16
                --mem-fraction-static 0.85
           e. configure_process_group (so kill takes down workers).
           f. Spawn child, capture stdout/stderr to tracing::debug.
           g. Health-poll loop:
                sleep(delay_ms); delay_ms = min(delay_ms × 2, 2000)
                if GET http://127.0.0.1:<port>/health → 200: BREAK
                if elapsed > startup_timeout_secs: kill child, return Err
           h. processes.insert(alias, SglangProcess { ... }).
           i. Return port.
```

The critical correctness invariant: **no `.await` runs while holding a DashMap shard lock**. `try_wait()` is sync, `AtomicI64::store` is sync — so the lock is released before any I/O happens.

llama.cpp's `get_or_spawn` is structurally identical but uses the `llama-server` binary from `<llama_cpp_dir>/build/bin/`.

---

## 9. Chat completion — request to streamed reply

A streaming OpenAI chat request is the hot path. Here's every step from client to first token:

```
                          CLIENT                                DAEMON                                BACKEND
                            │                                     │                                     │
1. POST /v1/chat/completions────────────────────────────────────► │                                     │
   { "model": "tiny", "messages": [...], "stream": true }         │                                     │
                            │                                     │                                     │
                            │       2. request_id_middleware                                            │
                            │          generate UUID, inject into extensions + tracing span             │
                            │                                     │                                     │
                            │       3. chat_completions(state, body: Bytes)                             │
                            │          metrics.chat_requests_total += 1                                 │
                            │                                     │                                     │
                            │       4. proxy_openai_chat_or_completion(...)                             │
                            │          Parse only "model" and "stream" from body (no full deserialize)  │
                            │                                     │                                     │
                            │       5. find_manifest(state, "tiny")                                     │
                            │          Lookup by alias → fallback to repo_id substring match            │
                            │                                     │                                     │
                            │       6. If manifest.modelfile.is_some():                                 │
                            │              apply_modelfile_to_body(body, mf, is_chat=true)              │
                            │                  - Prepend SYSTEM message                                 │
                            │                  - Insert seeded MESSAGEs                                 │
                            │                  - Fill in PARAMETER defaults (only where absent)         │
                            │                                     │                                     │
                            │       7. inference_router.get_endpoint(manifest)                          │
                            │          See section 8 above                                              │
                            │                                     │                                     │
                            │       8. proxy_bytes(state, url, body, is_stream=true)                    │
                            │          ───────────────────────────►                                     │
                            │                                     │  9. POST <backend>/v1/chat/completions
                            │                                     │     Content-Type: application/json  │
                            │                                     │     Accept: text/event-stream       │
                            │                                     │                                     │
                            │                                     │ ◄─── 10. HTTP 200, streaming body  │
                            │                                     │      Content-Type: text/event-stream│
                            │                                     │                                     │
                            │       11. Spawn relay task:         │                                     │
                            │           For each chunk from upstream:                                   │
                            │              metrics.bytes_proxied_total += chunk.len()                   │
                            │              tx.send(chunk).await                                         │
                            │              If client disconnected → break                               │
                            │                                     │                                     │
                            │       12. Response::builder()                                             │
                            │              .header("Content-Type", "text/event-stream")                 │
                            │              .header("Cache-Control", "no-cache")                         │
                            │              .body(ReceiverStream::new(rx))                               │
                            │                                     │                                     │
   ◄────────────────────────────────────────── 13. HTTP 200 + stream begins                             │
   X-Request-ID: <uuid>                                           │                                     │
                            │                                     │                                     │
   ◄────────────────────────── data: {"choices":[{"delta":{"content":"Hello"}}]}                        │
   ◄────────────────────────── data: {"choices":[{"delta":{"content":" world"}}]}                       │
   ◄────────────────────────── ...                                │                                     │
   ◄────────────────────────── data: [DONE]                       │                                     │
```

### Why this is fast

- **No body deserialize** on the proxy path. We parse only the `model` and `stream` fields with `serde_json::Value`. The rest of the body (potentially MBs of system prompts) is forwarded as raw bytes.
- **Zero-copy when no Modelfile**. The `Bytes` we receive from axum is passed through unchanged via reqwest's `.body(bytes)`.
- **Backpressure via bounded channel**. `mpsc::channel(256)` — if the client reads slowly, the channel fills and the upstream read pauses naturally.
- **Tuned connection pool**. `pool_idle_timeout=90s`, `pool_max_idle_per_host=32` so reused requests skip TCP+TLS handshake.

### Ollama variant

For `POST /api/chat` (Ollama-shaped), the daemon does extra translation work:

```
1. Deserialize OllamaChatRequest (full struct, not pass-through).
2. Apply Modelfile to messages array.
3. Build upstream OpenAI-style request JSON.
4. apply_modelfile_parameters() — Modelfile PARAMETER defaults.
5. apply_options() — request-level options win over Modelfile defaults.
6. proxy_to_ollama_ndjson() — same SSE relay as above, but on each
   `data: {...}` event, decode the OpenAI ChatCompletionResponse, extract the
   delta content, and re-emit it as Ollama's NDJSON shape:
       {"model":"tiny","created_at":"...","message":{"role":"assistant","content":"Hello"},"done":false}
   On `data: [DONE]`, emit a final frame with done:true.
```

This costs a tiny per-chunk deserialize+reserialize, but it's what lets Ollama clients talk to OpenAI-style backends transparently.

---

## 10. Modelfile pipeline

A Modelfile is an Ollama-style customization spec:

```
FROM llama-3.2-1b-instruct
SYSTEM "You are a terse expert. Answer in one short paragraph."
PARAMETER temperature 0.3
PARAMETER top_p 0.9
MESSAGE user "What is DNS?"
MESSAGE assistant "Domain Name System — maps names like example.com to IPs."
```

Pipeline:

```
1. localllm create my-bot -f Modelfile (or POST /api/create)
   │
   ▼
2. Modelfile::parse(source) — see registry/modelfile.rs
   - Walks each line, handles triple-quoted blocks.
   - Builds Modelfile { from, system, template, parameters, messages, source }.
   - Unknown directives logged at warn (graceful forward-compat).
   │
   ▼
3. Look up FROM target in the registry.
   │
   ▼
4. Clone base manifest. Set:
      new_manifest.alias = "my-bot"
      new_manifest.modelfile = Some(parsed)
      new_manifest.downloaded_at = now
   │
   ▼
5. registry.save(&new_manifest) — atomic temp-file rename, cache insert.
   │
   ▼
   Now "my-bot" is a routable model that points at the same weights as the base
   but carries SYSTEM/PARAMETER overrides.
```

At request time, the Modelfile is **applied per request**, not baked in:

```
POST /v1/chat/completions { "model": "my-bot", "messages": [{user, "Hi"}] }
   │
   ▼
apply_modelfile_to_body(body, mf, is_chat=true):
   │
   ├─ Inject SYSTEM (unless caller already provided one).
   ├─ Prepend seeded MESSAGEs.
   └─ Set PARAMETER defaults (only if caller didn't override).
   │
   ▼
Forward to inference backend, which sees:
   [
     {role: system,    content: "You are a terse expert..."},
     {role: user,      content: "What is DNS?"},
     {role: assistant, content: "Domain Name System — ..."},
     {role: user,      content: "Hi"}
   ]
   temperature: 0.3, top_p: 0.9
```

Two layered defaults: **Modelfile PARAMETER < request body**. The caller can always override.

---

## 11. Concurrency & locking model

localllm is heavily multithreaded — tokio runtime, axum handlers, multiple backend processes. Here's the locking story:

| Resource | Protection | Why |
|---|---|---|
| `ManifestStore.cache` | `Mutex<HashMap<...>>` | Updated rarely (pull/quantize/save), read often. Mutex is fine — contention is minimal. Poison-recovery wrapper means a panicked task doesn't take down the registry. |
| `SglangManager.processes` | `DashMap<String, SglangProcess>` | Per-alias sharding. No global lock on read or write. |
| `SglangProcess.last_used` | `Arc<AtomicI64>` | Touched on **every** inference request. An atomic store is ~1ns and lockless. |
| `SglangManager.port_alloc_lock` | `Mutex<()>` | Held only while probing ports during spawn. Never crosses an `.await`. |
| `LlamaCppManager` | Same pattern as sglang | Sister module, identical design. |
| Inference process child | `tokio::process::Child` owned by the DashMap entry | Lifetime tied to the entry; removed on `kill()`. |
| HTTP connection pool | `reqwest::Client` (Arc inside) | Cloned cheaply across handlers. |
| Metrics counters | `AtomicU64` | Lock-free increment. |

The critical correctness invariant: **no DashMap shard lock is held across an `.await` point**. This was a real hazard in earlier versions where a slow inference request could deadlock the registry. The fix: use `try_wait()` (sync) and `AtomicI64::store` (sync) inside the closure that holds the shard, defer all `.await` work to after the closure exits.

### Async vs sync

- `tokio::fs::*` everywhere for files (so file I/O doesn't block the runtime).
- `std::fs::*` only in `load_all` (deliberately blocking, runs in `spawn_blocking`).
- `tokio::process::Command` for child spawns (so `.wait()` doesn't block).
- `std::process::Command` for one-shot probes (`kill -0`, `tasklist`) where there's no benefit to async.

---

## 12. Background tasks

The daemon spawns these background tokio tasks at startup:

### 1. Manifest loader (one-shot)

```rust
tokio::task::spawn_blocking(move || registry_for_load.load_all());
```

Scans `manifests_dir`, loads every `*.json` into the cache. One-shot — runs to completion and exits.

### 2. TTL eviction loop (forever)

```
loop {
    sleep(60s)
    now = current Unix time
    
    for entry in sglang_manager.processes:
        last_used = entry.last_used.load()   // atomic
        if now - last_used > model_ttl_secs:
            kill candidates
    
    for alias in candidates:
        sglang_manager.kill(alias).await
    
    // Same for llama-cpp...
}
```

Runs forever. Default TTL is 5 minutes. The scan phase is entirely sync (atomic loads), so it can't deadlock. The kill phase is async but happens after the scan, with the alias list snapshotted.

### 3. Per-process stdout/stderr readers

For every spawned sglang and llama-server child, two tasks tail the pipes and forward each line to `tracing::debug`:

```rust
tokio::spawn(async move {
    let mut reader = tokio::io::BufReader::new(stdout).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        tracing::debug!("[sglang:{}:stdout] {}", alias, line);
    }
});
```

Without these, child pipes fill up and the child blocks on the next write. With them, output goes straight to the daemon's tracing subscriber.

### 4. Per-request relay tasks

For each in-flight chat/generate request, the proxy spawns a relay task that copies bytes from the upstream response stream into an mpsc channel feeding the client response body. These die when the request completes (either side disconnects).

---

## 13. Storage layout on disk

Default location: `~/.localllm/`. Override via `LOCALLLM_*_DIR` env vars.

```
~/.localllm/
├── daemon.pid                              # exclusive lock, removed on shutdown
├── config.toml                             # optional, all keys optional
│
├── manifests/                              # one JSON per model
│   ├── llama-3.2-1b-instruct.json
│   ├── my-bot.json                         # derived from FROM llama-3.2-...
│   └── tinyllama-1.1b-chat-v1.0.json
│
├── models/                                 # HuggingFace safetensors
│   └── meta-llama--Llama-3.2-1B-Instruct/  # repo_id with / → --
│       └── <revision_hash>/                # full commit SHA
│           ├── config.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── model.safetensors
│           └── ...
│
└── gguf/                                   # llama.cpp quantized outputs
    ├── llama-3.2-1b-instruct-Q4_K_M.gguf
    └── tinyllama-1.1b-chat-v1.0-Q4_K_M.gguf
```

Why two top-level dirs for model data?

- `models/` is "everything HuggingFace knows about" — the original artifacts.
- `gguf/` is "things we built from those originals" — derivative, can be regenerated.

This lets `/api/gc` distinguish orphaned source artifacts from orphaned derived ones, and lets users blow away `gguf/` to reclaim disk without losing the ability to re-quantize.

Manifest example:

```json
{
  "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
  "alias": "llama-3.2-1b-instruct",
  "revision": "abc123...",
  "local_path": "/home/user/.localllm/models/meta-llama--Llama-3.2-1B-Instruct/abc123...",
  "architecture": "llama",
  "weight_format": "Safetensors",
  "parameters_billion": 1.2,
  "context_length": 131072,
  "quantization": "Q4KM",
  "gguf_path": "/home/user/.localllm/gguf/llama-3.2-1b-instruct-Q4_K_M.gguf",
  "files": [
    {"name": "config.json", "sha256": "", "size_bytes": 877},
    {"name": "model.safetensors", "sha256": "", "size_bytes": 2471645608}
  ],
  "downloaded_at": "2026-05-17T10:30:00Z",
  "last_used":     "2026-05-17T10:42:11Z",
  "embeddings": false,
  "modelfile": null
}
```

---

## 14. Shutdown sequence

Triggered by Ctrl+C (any OS) or SIGTERM (Unix). The handler in [`daemon/server.rs`](src/daemon/server.rs):

```
1. tokio::signal::ctrl_c() OR signal(SIGTERM).recv() resolves.
2. axum's graceful_shutdown future completes.
3. axum::serve loop exits → stops accepting new connections.
4. In-flight requests drain (handlers complete naturally).
5. inference_router.kill_all().await:
       for each sglang alias → kill_process_tree (whole process group)
       for each llamacpp alias → same
       wait up to 10s per process for clean exit
6. drop(pid_lock) → removes ~/.localllm/daemon.pid.
7. main() returns.
```

`kill_process_tree` is what makes this clean across the process hierarchy:

- **Unix**: `kill -KILL -<pgid>` — negative PID means signal the whole group. Required because sglang's Python launcher forks worker subprocesses; killing only the direct child leaves workers orphaned.
- **Windows**: `taskkill /F /T /PID <pid>` — `/T` = "tree". Same reason.

If you Ctrl+C the daemon and check `nvidia-smi`, you should see VRAM freed within ~10 seconds. If you don't, that's a bug — file an issue.

---

## 15. Cross-platform behavior differences

| Concern | Windows | Unix |
|---|---|---|
| **Daemon detach** (auto-spawn) | `CreateProcess` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` (flags 0x08 | 0x200) | `fork → setsid → exec` so SIGHUP doesn't propagate from parent terminal |
| **Process group** for inference children | `CREATE_NEW_PROCESS_GROUP` (0x200) | `process_group(0)` (child becomes group leader) |
| **Process tree kill** | `taskkill /F /T /PID <pid>` | `kill -KILL -<pgid>` |
| **Process liveness probe** (PID lock) | `tasklist /FI "PID eq <pid>" /NH /FO CSV` | `kill -0 <pid>` |
| **Shutdown signals** | Only Ctrl+C | Ctrl+C + SIGTERM |
| **Default config paths** | `C:\Users\<you>\.localllm\` (dirs crate resolves) | `~/.localllm/` |
| **Binary path** | `target\release\localllm.exe` | `target/release/localllm` |
| **First-run engine build** (`setup.rs`) | `git clone` + `cmake --build` into `%USERPROFILE%\.localllm\llama.cpp`; resolves `llama-server.exe` | same, into `~/.localllm/llama.cpp`; resolves `llama-server` |
| **`install` target dir** | `%LOCALAPPDATA%\localllm\bin` | `~/.local/bin` |

The rest of the codebase is OS-agnostic — same Rust source, same tokio runtime, same axum router. The differences above are all confined to four small platform-specific blocks (one in main.rs, three in sglang.rs/server.rs).

---

## Appendix: file-by-file map

| File | Role |
|---|---|
| [`main.rs`](src/main.rs) | Entry point. Dispatches serve-vs-CLI, handles daemon auto-spawn. |
| [`config/settings.rs`](src/config/settings.rs) | `Settings` struct. Loads TOML + env overrides. |
| [`error.rs`](src/error.rs) | `LocalLlmError` enum. Most internals use `anyhow` instead. |
| [`cli/commands.rs`](src/cli/commands.rs) | Clap-derived CLI, one match arm per subcommand. |
| [`daemon/server.rs`](src/daemon/server.rs) | Daemon bootstrap, PID lock, shutdown handler. |
| [`api/routes.rs`](src/api/routes.rs) | All HTTP endpoint handlers, router builder, streaming proxies. |
| [`api/types.rs`](src/api/types.rs) | Wire types: OpenAI + Ollama + native. |
| [`api/middleware.rs`](src/api/middleware.rs) | Request-ID injection + tracing span. |
| [`registry/manifest.rs`](src/registry/manifest.rs) | `ModelManifest`, `ManifestStore`, safe-filename validation. |
| [`registry/modelfile.rs`](src/registry/modelfile.rs) | Modelfile parser and `apply_to_messages`. |
| [`downloader/hf_api.rs`](src/downloader/hf_api.rs) | HuggingFace API client (metadata + range-resume file download). |
| [`downloader/file_manager.rs`](src/downloader/file_manager.rs) | Pull orchestration, parallel downloads, architecture/param inference. |
| [`inference/mod.rs`](src/inference/mod.rs) | `InferenceRouter` — picks sglang vs llama.cpp. |
| [`inference/sglang.rs`](src/inference/sglang.rs) | sglang process manager + cross-OS process-group helpers. |
| [`inference/llamacpp.rs`](src/inference/llamacpp.rs) | llama-server process manager (mirrors sglang). |
| [`compression/quantize.rs`](src/compression/quantize.rs) | Two-stage convert + quantize pipeline. |
| [`gpu/vram.rs`](src/gpu/vram.rs) | nvidia-smi parser, VRAM fit estimator. |
