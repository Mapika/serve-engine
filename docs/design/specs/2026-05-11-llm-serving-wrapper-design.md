# Service Router - Design

**Status:** Draft, brainstorming output
**Date:** 2026-05-11
**Working name:** `serve-engine` (final TBD)

## 1. Problem & audience

vLLM, SGLang, and TensorRT-LLM are best-in-class inference engines, but their operator UX is hostile: each instance hosts a single model, swapping models means restarting the server, configuration is YAML/CLI-flag heavy, install is dependency-conflict hell, and there is no built-in concept of multi-tenant API keys, fair queueing, or a coherent admin UI. The result is that everyone running a serious GPU box ends up writing the same orchestration scaffolding badly.

The target user is **anyone who owns one beefy GPU box and wants to serve their own users from it.** That spans:

- Homelabbers with a 4090 / 4090 / H100 serving themselves and a few friends or a Discord bot.
- Engineering organisations that buy a single 8x B300 instance and serve the whole company from it.

The unifying property is **single-node, self-hosted, multi-user**. Not SaaS, not multi-node distributed serving. The product is "the inference server you put on your one big GPU box, and it handles lifecycle and routing without extra app code."

Speed and concurrency are not our problem - vLLM and SGLang already solve them via continuous batching, paged/radix attention, etc. Our job is orchestration, lifecycle, and UX.

## 2. Goals & non-goals

### Goals
- **Multi-model on one daemon:** one process hosts a registry of models, swaps them on demand, exposes one OpenAI-compatible HTTP API.
- **Pinned + auto-swap lifecycle:** user pins certain models as always-loaded; others get LRU/idle eviction.
- **Zero-config / autotune:** detect GPUs, pick the right engine, TP size, KV cache budget, dtype, and feature flags automatically.
- **Fast loading:** OS page cache + engine fast-load flags + parallel pre-warm of pinned models on startup.
- **Easy install:** one command (`uv tool install serve-engine`) or one Docker pull; one diagnostic (`serve doctor`) for anything that's wrong.
- **GPU topology awareness:** placement decisions respect NVLink connectivity and tensor-parallel constraints, not just a flat VRAM number.
- **Multi-tenancy within the box:** API keys (OpenAI-compatible `sk-` prefix), per-key rate limits, fair queueing under contention.
- **Tasteful web UI:** dashboard, models, playground, API keys, logs.
- **Use what the engines provide:** pass-through advanced features (response_format, tools, LoRA, speculative decoding) transparently; never reimplement an inference primitive.

### Non-goals (v1)
- Multi-node distributed serving.
- Fine-tuning or training.
- Custom model formats beyond what the engines support.
- Built-in billing / invoicing.
- Predictive prefetch, custom weight snapshotting, cross-engine weight sharing.
- TensorRT-LLM (planned for v1.1; engine abstraction designed to admit it).
- Multi-org / multi-team RBAC (single admin + many API keys is the v1 model).
- Audit logging beyond structured stdout.

## 3. Architecture

### 3.1 Components

1. **Daemon** - long-running Python process. Owns model registry, lifecycle manager, request router, OpenAI-compatible API, web UI, admin API, metrics. Started by `serve daemon start`, systemd, or run as a container.
2. **CLI** (`serve`) - thin Python client. Talks to the daemon over a Unix domain socket at `~/.serve/sock`. Does not import vLLM / SGLang; starts fast.
3. **Engine containers** - one per loaded model. Use upstream official images (`vllm/vllm-openai:<tag>`, `lmsysorg/sglang:<tag>`). The daemon spawns and supervises them through the Docker API.
4. **Web UI** - Vite + React SPA bundled into the Python package, served by the daemon at `http://localhost:<port>/`. Reads admin API; no separate backend-for-frontend.
5. **Local model store** - `~/.serve/models/` following HF Hub's snapshot layout so the cache is shared with other tools (transformers, native vLLM).
6. **State store** - SQLite at `~/.serve/db.sqlite` for the model registry, deployments, API keys, usage, configuration.

### 3.2 Process & container topology

- Daemon process binds to the configured public port (default `11500`) and to the local Unix socket.
- Engine containers join a dedicated Docker bridge network `serve-engines` and are addressed by container name from the daemon. They do **not** map ports to the host; only the daemon does.
- Daemon runs on the host (recommended; installed with `uv tool install`) or as its own container with a bind-mounted Docker socket. Either way, lifecycle of engine containers is daemon-driven, not declarative.

### 3.3 Request flow (inference)

```
client --POST /v1/chat/completions--> daemon
                                        | 1. authenticate API key, check quota
                                        | 2. parse `model` field
                                        | 3. lifecycle manager:
                                        |      loaded? -> route
                                        |      not loaded? -> swap, wait, route
                                        v
                                   engine container (serve-engines/<id>:<port>)
                                        |
client <-----stream proxied back--- daemon
```

Streaming is end-to-end - daemon does not buffer.

### 3.4 Control flow (CLI / Web UI)

CLI hits the Unix socket; Web UI hits `/admin/*` on the public port with the admin API key. Both call the same handlers. CLI commands stream progress (downloads, loads) back over the socket.

## 4. Model lifecycle

### 4.1 Deployment abstraction

A *deployment* is `(model, gpu_set, backend, image_tag, engine_args, status)`. One model may have multiple deployments (different TP shapes, different engines). State lives in SQLite. Status transitions: `pending -> loading -> ready -> stopping -> stopped` (with `ready` and serving requests being the same state), plus `failed` as a terminal state reachable from any active state.

### 4.2 Model registry & pull

- Model entry: `(name, hf_repo, revision, local_path, default_backend, default_engine_args, aliases)`.
- `serve pull <name>` resolves alias -> HF repo -> downloads via `huggingface_hub` (parallel chunks, resume, checksum) to `~/.serve/models/`.
- Built-in catalog (a YAML shipped with the package) covers popular models with sensible defaults. Users add custom entries with `serve model add`.

### 4.3 GPU placement

- On startup, daemon enumerates GPUs via `pynvml` and reads the topology matrix from `nvidia-smi topo -m`.
- Per-GPU state: total / allocated memory, current deployment owner, NVLink-island membership.
- Placement constraints: tensor-parallel size must be a power of 2 and must divide `num_attention_heads`; the selected GPU set must be NVLink-connected for any TP > 1 (warn-only fallback if not).
- Placement algorithm for a new deployment:
  1. Try to fit on currently-free GPUs.
  2. If not, find the minimum-disruption set of auto (non-pinned) deployments to evict.
  3. If still not, refuse with a clear "no room" error and a suggested action (unpin a specific deployment, or load with smaller TP / context).
- Prefers leaving the largest contiguous NVLink island free for future medium-sized models.

### 4.4 Pin / auto-swap / idle eviction

- Each deployment is `pinned` or `auto`. Pinned never evicts automatically.
- Auto deployments have an `idle_timeout` (default 5 min, configurable per-model). A background reaper evicts ones that have gone idle.
- Request for a non-loaded model triggers placement -> eviction (if needed) -> `docker run` for the new engine -> wait for `/health` -> route. Caller holds during the swap with a generous deadline (5 min default for big models).
- Lifecycle transitions emit structured events on `/admin/events` (SSE) so the UI stays coherent.

### 4.5 Fast loading

For v1, three pragmatic wins:

1. OS page cache for weights (mmap by both engines; we don't fight it).
2. Pre-warm of pinned models on daemon startup, parallel up to GPU/IO budget.
3. Pass-through of engine fast-load flags (vLLM's `--load-format=runai_streamer` etc.) via the autotune layer.

Explicitly not in v1: predictive prefetch, custom snapshotting, shared-memory weight pools.

### 4.6 Failure handling

- Engine container exits unexpectedly -> detected via health-check loop and Docker events. Auto-restart up to 3 times with exponential backoff. After that, deployment goes `failed`; new requests return a clean 503 with the engine's last log lines.
- OOM at load time -> if autotune is enabled, retry once with reduced `gpu-memory-utilization`. Otherwise surface the engine's own error.
- Load deadline exceeded -> kill the container, return a clear timeout, surface the relevant tail of engine logs.

The throughline: **every failure produces structured logs, a state event, and an actionable HTTP response.**

## 5. User-facing surfaces

### 5.1 CLI

```
serve pull <model>             # download + register
serve run <model>              # pull (if needed) + load + REPL
serve ls                       # registered models
serve ps                       # active deployments + per-GPU stats
serve pin <model> | unpin <model>
serve rm <model>               # delete weights + registry entry
serve logs <model> [-f]        # tail engine logs
serve top                      # live htop-style view of GPUs / deployments
serve daemon {start|stop|status|restart}
serve key {create|list|revoke}
serve config {show|set <k> <v>}
serve update-engines [<backend>]
serve pull-engine <backend>
serve doctor                   # diagnose CUDA, drivers, Docker, NVLink, ports
serve setup                    # interactive first-run wizard
```

All commands support `--json` for scripting.

### 5.2 HTTP API

Three slices on the same port:

**(a) OpenAI-compatible - the inference product surface:**
- `POST /v1/chat/completions` (streaming + non-streaming)
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/responses`
- `GET /v1/models`

Advanced request fields (`response_format`, `tools`, `logit_bias`, LoRA via `model: "base@adapter"` syntax) ride through transparently to the engine. We do not validate or reshape them.

**(b) Engine-native passthrough - power-user escape hatch:**
- `POST /engines/{deployment_id}/*` - proxied directly to the engine container.

Any engine feature shipped upstream is usable on day one.

**(c) Admin / control plane - for CLI and Web UI:**
- `GET|POST|DELETE /admin/deployments[/:id][/pin]`
- `GET|POST|DELETE /admin/models[/:id]`
- `GET|POST|DELETE /admin/keys[/:id]`
- `GET /admin/gpus` - topology + live utilization
- `GET /admin/events` - SSE lifecycle event stream
- `GET /metrics` - Prometheus, aggregated from engine `/metrics`

Auth: API keys with `sk-` prefix on `/v1/*`; an admin key (auto-generated on first run, printed to CLI) on `/admin/*`.

### 5.3 Web UI

Vite + React SPA bundled in the package, served at `/`. Five screens:

1. **Dashboard** - loaded models, per-GPU memory bars (real device topology rendered), live throughput sparkline.
2. **Models** - registry list, pull with progress, pin/unpin toggles, "load now" button.
3. **Playground** - simple chat with model picker.
4. **API Keys** - create / revoke / per-key usage stats.
5. **Logs** - tailed engine logs per deployment.

No user accounts beyond API keys; no role-based permissions; no multi-org tenancy.

## 6. Engine abstraction & autotune

### 6.1 Backend interface (thin)

```python
class Backend(Protocol):
    name: str                       # "vllm" | "sglang"
    image_default: str              # e.g. "vllm/vllm-openai:v0.7.3"
    package_for_doctor: str | None  # name to mention if missing
    health_path: str = "/health"
    openai_base: str = "/v1"
    metrics_path: str = "/metrics"

    def build_argv(self, plan: DeploymentPlan) -> list[str]: ...
    def container_env(self, plan: DeploymentPlan) -> dict[str, str]: ...
    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]: ...  # gpus, ipc, shm_size, ulimits
    def parse_progress(self, line: str) -> LoadProgress | None: ...
```

This is the entire contract. ~80% of integration work per backend is `build_argv`. Adding TRT-LLM later is one more `Backend` plus an adapter for its build-time compile step.

We do **not** abstract over inference primitives - OpenAI requests go to the engine unmodified.

### 6.2 DeploymentPlan (autotune output)

```
DeploymentPlan {
  model_id: str               # resolved HF repo + revision
  backend: "vllm" | "sglang"
  image_tag: str
  gpu_ids: list[int]
  tensor_parallel: int        # = len(gpu_ids) by default
  pipeline_parallel: int      # usually 1
  dtype: "auto" | "bf16" | "fp16" | "fp8"
  quantization: None | "awq" | "gptq" | "fp8" | "marlin"
  max_model_len: int
  gpu_memory_utilization: float
  enable_prefix_caching: bool
  enable_chunked_prefill: bool
  speculative: None | SpecDecodeConfig
  extra_args: dict[str, str]
}
```

### 6.3 Autotune algorithm

Inputs: model metadata (params, dtype, head config, max context), available GPU set, user prefs (`--latency` / `--throughput`, `--engine`, `--ctx`).

1. **Engine choice.** Default vLLM. Switch to SGLang for known prefix-caching beneficiaries (agentic, multi-turn-heavy) and certain MoE families (Deepseek-V3, Qwen variants benefiting from DP-attention). Decisions live in `backends/selection.yaml`, not in code.
2. **Quantization.** Inherit from checkpoint if present; else `bf16` on Ampere+ Blackwell, `fp16` on older. No on-the-fly quantization in v1.
3. **TP/PP sizing.** Smallest power-of-2 TP that satisfies: `weights/TP + KV_cache_at_target_concurrency + activations < per_gpu_vram x 0.9`. Must divide `num_attention_heads`. Refuse with a clear message if no TP fits.
4. **GPU placement.** Section 4.3 algorithm against current state.
5. **Context length.** Clamp user value to model max; default to `min(model_max, 32k)`.
6. **gpu-memory-utilization.** 0.9 single-tenant; scaled down when sharing GPUs.
7. **Feature flags.** `enable_prefix_caching` and `enable_chunked_prefill` on by default (free wins).
8. **Speculative decoding.** Off by default; opt-in via flag.

Every chosen value is logged before engine start with rationale.

### 6.4 Backend manifest

`backends.yaml` shipped with the package:

```yaml
vllm:
  image: vllm/vllm-openai
  pinned_tag: v0.7.3
  health_path: /health
  openai_base: /v1
  metrics_path: /metrics
sglang:
  image: lmsysorg/sglang
  pinned_tag: v0.4.2.post1
  ...
```

Overridable: `serve config set engine.vllm.image vllm/vllm-openai:v0.7.5`. `serve update-engines` re-pins to current upstream-tested versions; release notes call out which engines are blessed per our release.

## 7. Concurrency, fairness, and the "company case"

The Python orchestrator is not the bottleneck - engine batching is. Our job is to stay out of the way.

- Daemon is fully async (FastAPI + uvicorn; httpx async client with a generous connection pool to each engine). Streaming requests pass through without buffering.
- We do **not** serialise requests; engines receive concurrent in-flight requests so their internal schedulers do the batching.
- A bounded number of concurrent requests per engine prevents the engine's scheduler queue from growing pathologically.
- Beyond engine capacity, requests queue at the orchestrator with weighted-fair queueing per API key, so one heavy user can't starve others.
- Per-API-key rate limits (token / request / per-window) configured at key creation.

The 8x B300 case (~640 GB HBM) typically hosts a mix: one TP-8 frontier model, or several smaller models on disjoint GPU sets. Placement handles both.

## 8. Packaging, install, observability, testing

### 8.1 Install paths

1. **`uv tool install serve-engine`** (recommended for individuals / homelabbers). One command, fast lockfile, no global Python pollution.
2. **Daemon as a container** (`ghcr.io/<org>/serve-engine:latest`) - bind-mount Docker socket + `~/.serve`. Recommended when teams don't want Python on the host at all.
3. **`serve install-service`** - writes a systemd unit referencing the `uv tool install`-ed binary. For "survive reboots" production.
4. **`curl -sSL https://<host>/install.sh | sh`** - wraps option 1 with environment checks (Docker, nvidia-container-toolkit) and runs `serve setup` at the end.

### 8.2 Observability

- Structured JSON logs to stdout and `~/.serve/logs/daemon.log` (rotated).
- `/metrics` in Prometheus format, aggregated across engines: per-deployment latency percentiles, throughput, KV cache utilization; per-GPU utilization / memory / power; per-API-key usage.
- `/admin/events` SSE stream powers the dashboard and CLI `serve top`.
- No built-in tracing system. OTEL env vars pass through to engine containers if users configure them.

### 8.3 Testing strategy

1. **Unit (Python, no GPU, <30s):** autotune decisions, placement, lifecycle state machine, OpenAI translation, fair queueing.
2. **Integration (containerised fake engine, no GPU):** end-to-end daemon behavior - slow loads, OOM, crash mid-stream, slow streams. Runs in CI.
3. **GPU smoke (real hardware, self-hosted runner):** pull Llama-3.2-1B, load, run 100 concurrent chats, verify p99 below target. Triggered manually or pre-release.

### 8.4 Diagnostic - `serve doctor`

```
OK CUDA 12.6 detected (driver 565.x)
OK 8 GPUs visible (8x B300, NVLink mesh)
OK Docker 26.1 running, current user in docker group
OK nvidia-container-toolkit configured (cuda:12.4-base passes nvidia-smi)
OK vllm/vllm-openai:v0.7.3 pulled
FAIL lmsysorg/sglang:v0.4.2.post1 missing -> `serve pull-engine sglang`
OK Port 11500 free
OK ~/.serve writable
OK HF_TOKEN set (gated models accessible)
```

Principle for the whole product: **every config decision the user could be forced to make is either auto-detected, has a sensible default, or has a `doctor` line telling them how to fix it.**

## 9. Repository structure (proposed)

```
serving-engine/
|-- pyproject.toml
|-- README.md
|-- src/serve_engine/
|   |-- __init__.py
|   |-- daemon/                  # FastAPI app, lifecycle manager, router
|   |-- cli/                     # Click / Typer commands (thin)
|   |-- backends/                # vllm.py, sglang.py, selection.yaml, backends.yaml
|   |-- lifecycle/               # placement, eviction, deployment state machine
|   |-- autotune/                # plan builder, heuristics
|   |-- store/                   # SQLite models, migrations
|   |-- proxy/                   # OpenAI translation, streaming, queueing
|   |-- admin_api/               # /admin/*
|   |-- observability/           # logs, metrics, events
|   |-- doctor/                  # environment checks
|   +-- ui/                      # bundled SPA assets
|-- ui/                          # source for the SPA (Vite + React)
|-- docker/
|   |-- daemon.Dockerfile        # for the containerised daemon option
|   +-- fake-engine.Dockerfile   # test double
+-- tests/
    |-- unit/
    |-- integration/
    +-- smoke/
```

## 10. Open questions / future work

- Final naming (CLI binary, package, repo, default daemon port).
- TRT-LLM integration shape - its build-time engine compilation needs an extra step before the deployment can be marked "ready"; design admits this but doesn't pin the UX yet.
- Speculative-decoding UX - the engines support several modes (EAGLE, n-gram, MTP); decide how to expose.
- Whether to ship a default-deny outbound network policy on engine containers (we mount the model cache, so engines don't *need* outbound; this is a hardening hook for company deployments).
- Whether `serve update-engines` should also bump the daemon (probably not - daemon and engines update on separate cadences).
