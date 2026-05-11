# serve-engine

A single-node, multi-user inference orchestrator over vLLM (and soon SGLang).

Goal: solve the operator UX gap left by `vllm serve` / `python -m sglang.launch_server` — one daemon, multi-model, OpenAI-compatible, container-isolated engines, no YAML.

This repository is in active early development. Plan 01 (the walking skeleton) ships single-model serving over vLLM.

## Requirements

- Linux with NVIDIA GPU(s)
- Docker 24+ with `nvidia-container-toolkit`
- Python 3.11+ and `uv`

## Quickstart

```bash
# Install
uv tool install -e .

# Start the daemon
serve daemon start

# Register a model
serve pull meta-llama/Llama-3.2-1B-Instruct --name llama-1b

# Load it on GPU 0
serve run llama-1b --gpu 0

# Talk to it (any OpenAI client works)
curl http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-1b","messages":[{"role":"user","content":"Hello"}]}'

# Stop it
serve stop
serve daemon stop
```

## Plan 01 limitations (intentional)

- One model at a time. Loading a new model stops the current one.
- vLLM only. SGLang comes in Plan 04.
- No authentication. The TCP port is open to anyone on the host; bind to `127.0.0.1` (default) and use a reverse proxy if exposing externally.
- No autotune. You pass `--gpu`, `--ctx`, `--dtype` yourself.
- No web UI.

See `docs/superpowers/specs/` for the full design and `docs/superpowers/plans/` for the planned slices.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
```
