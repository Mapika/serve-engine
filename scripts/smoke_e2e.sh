#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for Plan 01 walking skeleton.
#
# Prerequisites:
#   - Docker daemon running
#   - nvidia-container-toolkit configured
#   - At least one CUDA-capable GPU (id 0)
#   - HuggingFace cache primed, or HF_TOKEN set if the model is gated
#   - `uv pip install -e ".[dev]"` already done in an active venv
#
# Verifies:
#   - daemon start/stop
#   - pull (register)
#   - run (download + container spawn + health-wait)
#   - /v1/chat/completions returns a streamed body containing the prompt cue
#   - stop tears down the container

cleanup() {
    serve stop 2>/dev/null || true
    serve daemon stop 2>/dev/null || true
}
trap cleanup EXIT

serve daemon start

serve pull meta-llama/Llama-3.2-1B-Instruct --name llama-1b

serve run llama-1b --gpu 0 --ctx 8192 --dtype auto

echo "Hitting /v1/chat/completions ..."
curl -sS -N http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-1b",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "stream": true,
    "max_tokens": 4
  }' | tee /tmp/serve_smoke.out

if ! grep -q "OK" /tmp/serve_smoke.out; then
    echo "FAIL: expected 'OK' in streamed response"
    exit 1
fi

echo "PASS"
