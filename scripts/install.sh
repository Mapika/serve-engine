#!/usr/bin/env bash
set -euo pipefail

# serve-engine bootstrap installer.
#
# Usage:
#   curl -fsSL https://example.com/install.sh | bash
#
# What it does:
#   1. Installs `uv` if missing (https://docs.astral.sh/uv/).
#   2. `uv tool install` the `serve-engine` package (or `pip install -e .` if run inside a checkout).
#   3. Runs `serve doctor`.
#   4. Prints next steps.

REPO_DIR=""
if [ -f "pyproject.toml" ] && grep -q "serve-engine" pyproject.toml 2>/dev/null; then
    REPO_DIR="$(pwd)"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo ">>> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    [ -f "$HOME/.local/share/uv/env" ] && . "$HOME/.local/share/uv/env" || true
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">>> uv $(uv --version)"

if [ -n "$REPO_DIR" ]; then
    echo ">>> installing serve-engine from local checkout: $REPO_DIR"
    uv tool install --editable "$REPO_DIR"
else
    echo ">>> installing serve-engine from PyPI (or remote)"
    uv tool install serve-engine
fi

echo
echo ">>> running serve doctor"
if serve doctor; then
    echo
    echo "✓ environment looks good. Next:"
    echo
    echo "    serve setup        # interactive wizard (recommended)"
    echo "    # or:"
    echo "    serve daemon start"
    echo "    serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b"
    echo "    serve run qwen-0_5b --gpu 0"
else
    echo
    echo "! serve doctor reported issues. Fix them and re-run \`serve doctor\`."
    exit 1
fi
