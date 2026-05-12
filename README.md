# serve-engine

A single-node, multi-user LLM inference orchestrator over vLLM, SGLang, and TensorRT-LLM.

`serve-engine` solves the operator-UX gap left by `vllm serve` / `python -m sglang.launch_server` / `trtllm-serve`: one daemon, multiple models, pin / auto-swap lifecycle, OpenAI-compatible HTTP, API keys with proper rate limits, a small web UI, and live observability — on one GPU box.

**Status:** core lifecycle, observability, auth, and three engine backends (vLLM, SGLang, TensorRT-LLM) verified end-to-end on H100 and RTX PRO 6000 Blackwell. 156 unit + integration tests, ruff clean.

## What it does

- **One daemon, many models.** Register N models, pin some, let the rest auto-swap on demand. KV-aware GPU placement picks where each model lands.
- **Engine pluggability.** Same `/v1/chat/completions` API regardless of whether the model is served by vLLM, SGLang, or TensorRT-LLM. Engine choice is per-model (`--engine vllm|sglang|trtllm`) or YAML-driven by pattern (`backends/selection.yaml`). Auto target-concurrency picks `--max-num-seqs` / `--max_batch_size` from model architecture so small models don't get the conservative 30B-class default.
- **OpenAI-compatible.** Drop-in: anything that speaks the OpenAI API (Python SDK, JS SDK, LangChain, `curl`) just works. Auth via `Authorization: Bearer sk-...`.
- **Real rate limits.** Eight-window sliding limiter (RPM, RPH, RPD, RPW × tokens and requests), per-tier defaults + per-key overrides, returns `429` with `Retry-After`.
- **Crash-safe.** Graceful shutdown stops engines cleanly; startup reconciliation re-adopts surviving containers or marks orphans failed.
- **Observable.** Prometheus `/metrics` aggregated across engines, `/admin/events` SSE for lifecycle transitions, `/admin/gpus` for live per-GPU stats, `serve top` for a terminal dashboard.
- **Web UI.** Five screens (dashboard, models, playground, keys, logs) bundled into the Python package — no separate frontend deploy.
- **Bootstrap-friendly.** `serve doctor` diagnoses the environment; `serve setup` does the interactive first-run.

## Requirements

- Linux + an NVIDIA GPU (single-node; no multi-host support)
- Docker 24+ with GPU access (the daemon spawns engine containers via the Docker API)
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (recommended) or pip

## Install

### Option 1 — `uv tool install` (recommended)

```bash
git clone https://github.com/Mapika/serve-engine
cd serve-engine
uv tool install --editable .
serve doctor
serve setup        # interactive wizard
```

### Option 2 — daemon-as-container

```bash
git clone https://github.com/Mapika/serve-engine
cd serve-engine
docker build -f docker/daemon.Dockerfile -t serve-engine:dev .
docker run -d --name serve \
  --network host \
  -v ~/.serve:/root/.serve \
  -v /var/run/docker.sock:/var/run/docker.sock \
  serve-engine:dev
```

## Quickstart

```bash
serve daemon start
serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b   # registers AND downloads
serve run qwen-0_5b --gpu 0 --pin                        # pinned (never auto-evict)
serve pull Qwen/Qwen2.5-1.5B-Instruct --name qwen-1_5b
serve run qwen-1_5b --gpu 0 --idle-timeout 60            # auto-evict after 60s idle

curl http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-0_5b","messages":[{"role":"user","content":"Hello"}]}'
```

For the web UI: `serve key create web --tier admin` (creates the first admin key over the local UDS, no auth needed), then open `http://127.0.0.1:11500/` and paste the secret.

## CLI

```
serve doctor              # check environment
serve setup               # interactive first-run wizard
serve daemon {start|stop|status}
serve pull <repo>         # register + download weights
serve ls                  # list registered models
serve run <name>          # load a model
serve pin <name>          # never auto-evict
serve unpin <name>        # mark as LRU-evictable
serve ps                  # list deployments
serve stop [<id>]         # stop one or all
serve top                 # live dashboard
serve logs                # tail engine container
serve key {create|list|revoke}
serve update-engines      # check Docker Hub for newer pinned tags
```

## Architecture

```
                       ┌──────────────────────────────────────┐
                       │           serve daemon               │
                       │  (FastAPI + uvicorn, async)          │
                       │                                      │
   client ──TCP──▶─────┤  /v1/chat/completions ─────┐         │
                       │  /v1/completions            │         │
                       │  /v1/embeddings             │         │
                       │  /v1/models                 │         │
                       │  /admin/*  (Bearer / UDS)   │         │
                       │  /metrics                   │         │
                       │  /admin/events  (SSE)       │         │
                       │  /admin/gpus                │         │
                       │  /  (web UI)                │         │
                       │                             ▼         │
   CLI ───UDS sock─────┤  LifecycleManager  ──┐  proxy router  │
                       │  ↑   ↓               │     │          │
                       │  ↑   reconcile/stop  │     │          │
                       │  EventBus → SSE      │     │          │
                       └───────────┼──────────┼─────┼──────────┘
                                   │          │     │
                                   ▼          ▼     ▼
                          ┌────────────────────────────────┐
                          │   Docker API (host socket)     │
                          └────────────────────────────────┘
                                   │
                                   ▼
                          ┌────────────────────────────────┐
                          │  Engine containers (one/model) │
                          │    vllm/vllm-openai            │
                          │    lmsysorg/sglang             │
                          │  bound to 127.0.0.1:<random>   │
                          └────────────────────────────────┘
```

- **Daemon** runs on the host, single Python process, async end-to-end.
- **Engines** run as separate containers via the Docker API. Daemon talks to them over a host-bound random port (no shared-network hop, no DNS surprises).
- **State** lives in SQLite at `~/.serve/db.sqlite` (models, deployments, API keys, usage).
- **Backend abstraction**: each engine is a `Backend` Protocol implementation; argv differences (`--model` vs `--model-path`, `--gpu-memory-utilization` vs `--mem-fraction-static`) live entirely inside the engine class.
- **Manifest-driven**: image tags, internal ports, and engine-specific headroom constants come from `src/serve_engine/backends/backends.yaml`. Override per-host in `~/.serve/backends.override.yaml`.

## Tested performance

Single H100 80GB, Qwen2.5 0.5B and 1.5B, 512-token outputs, Poisson arrivals (raw JSON in `docs/bench/`):

| QPS | Model/Engine | Agg TPS | TTFT p50 (ms) | E2E p50 (ms) |
|---:|---|---:|---:|---:|
| 1 | 0.5B/vllm | 355 | 25 | 1134 |
| 16 | 0.5B/vllm | 7 169 | 33 | 1429 |
| 32 | 0.5B/sglang | **14 751** | 68 | 1280 |
| 16 | 1.5B/sglang | 7 904 | 38 | 1608 |
| 32 | 1.5B/vllm | 13 377 | 128 | 2814 |

Engine saturation on the small Qwens is ~14k tokens/sec aggregate.

## Design docs and plans

The implementation was built in seven stacked plans (`docs/superpowers/plans/`) plus an initial design (`docs/superpowers/specs/`). Each plan was verified live on an H100 before the next one started.

- Plan 01 — walking skeleton (daemon + CLI + vLLM container)
- Plan 02 — multi-model lifecycle (pin / auto-swap / KV-aware placement)
- Plan 03 — SGLang backend
- Plan 04 — API keys + multi-window rate limits
- Plan 05 — observability (`/metrics`, SSE events, `serve top`)
- Plan 06 — web UI (Vite + React + Tailwind, bundled)
- Plan 07 — `serve doctor` + installer + daemon-as-container
- Plan 08 — hardening pass (crash recovery, proxy status forwarding, real `pull` download, etc.)

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                    # 131 tests, ~12s
ruff check src/ tests/

# UI dev (optional — repo ships pre-built dist)
cd ui && npm install && npm run build
```

## Out of scope (v1)

- Multi-node distributed serving (you have ≥8 GPUs and want tensor-parallel across hosts → use vLLM directly).
- Fine-tuning / training (inference-only).
- Autotune (model → optimal TP / dtype / context auto-pick). The KV estimator + manifest-driven headroom does most of the painful tuning; full autotune is parked.
- Built-in TLS. Bind to `127.0.0.1` and put a reverse proxy in front for external access.

## License

TBD.
