# Serving Engine — Plan 07: Doctor + Installer + Daemon-as-Container

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Make installation and diagnosis trivial. Three deliverables:

1. **`serve doctor`** — top-down environment check: CUDA driver, Docker, nvidia-container-toolkit, GPU enumeration, NVLink topology, port availability, `~/.serve` writability, HF_TOKEN, cached engine images, daemon status.
2. **`serve setup`** — interactive first-run wizard: confirms doctor passes, creates the initial admin key, prints the web UI URL.
3. **`install.sh`** — one-shot bootstrap: installs `uv` if missing, `uv tool install` the package, runs `serve doctor`, prints next steps. Lives at `scripts/install.sh`.
4. **Daemon-as-container** — a `docker/daemon.Dockerfile` and `docker/daemon-compose.yml` for users who prefer Docker over a host install.

No new third-party libraries. Pure stdlib + `pynvml` (already a dep).

---

## File structure

```
serving-engine/
├── src/serve_engine/
│   ├── doctor/
│   │   ├── __init__.py
│   │   ├── checks.py             # NEW — each check is a function → CheckResult
│   │   └── runner.py             # NEW — runs all checks, formats results
│   ├── cli/
│   │   ├── doctor_cmd.py         # NEW
│   │   └── setup_cmd.py          # NEW
│   └── ...
├── docker/
│   ├── daemon.Dockerfile         # NEW
│   └── README.md                 # NEW
├── scripts/
│   └── install.sh                # NEW
└── tests/
    └── unit/
        └── test_doctor.py        # NEW
```

---

## Task 1: Doctor check primitives

**Files:** `src/serve_engine/doctor/__init__.py` (empty), `src/serve_engine/doctor/checks.py`, `tests/unit/test_doctor.py`

A check returns `CheckResult(name, status, detail, fix)` where `status ∈ {"ok","warn","fail"}`.

- [ ] **Tests**

```python
from serve_engine.doctor.checks import (
    CheckResult,
    check_docker,
    check_gpus,
    check_paths,
    check_ports,
)


def test_check_paths_writable(tmp_path, monkeypatch):
    monkeypatch.setattr("serve_engine.doctor.checks.SERVE_DIR", tmp_path)
    r = check_paths()
    assert r.status == "ok"
    assert "writable" in r.detail.lower()


def test_check_paths_not_writable(tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    bad.chmod(0o400)  # read-only
    monkeypatch.setattr("serve_engine.doctor.checks.SERVE_DIR", bad)
    r = check_paths()
    assert r.status in ("warn", "fail")
    bad.chmod(0o755)  # restore for cleanup


def test_check_ports_free(monkeypatch):
    monkeypatch.setattr("serve_engine.doctor.checks.DEFAULT_PORT", 0)
    r = check_ports()
    # Port 0 always binds; check returns ok
    assert r.status == "ok"


def test_check_docker_unreachable(monkeypatch):
    def fake_docker_from_env():
        raise RuntimeError("connection refused")
    monkeypatch.setattr("serve_engine.doctor.checks._docker_from_env", fake_docker_from_env)
    r = check_docker()
    assert r.status == "fail"
    assert "docker" in r.detail.lower()


def test_check_gpus_no_pynvml(monkeypatch):
    monkeypatch.setattr("serve_engine.doctor.checks.pynvml", None)
    r = check_gpus()
    assert r.status == "fail"
    assert "pynvml" in r.detail.lower() or "no" in r.detail.lower()
```

- [ ] **Implement `src/serve_engine/doctor/checks.py`**

```python
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from serve_engine.config import DEFAULT_PUBLIC_PORT, SERVE_DIR

try:
    import pynvml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]

try:
    import docker  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    docker = None  # type: ignore[assignment]


# Module-level for monkeypatching in tests.
DEFAULT_PORT = DEFAULT_PUBLIC_PORT


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    fix: str | None = None


def _docker_from_env():
    if docker is None:
        raise RuntimeError("docker SDK not installed")
    return docker.from_env()


def check_paths() -> CheckResult:
    p = Path(SERVE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    if not os.access(p, os.W_OK):
        return CheckResult(
            name="serve directory",
            status="fail",
            detail=f"{p} is not writable",
            fix=f"chmod u+w {p}",
        )
    return CheckResult(name="serve directory", status="ok", detail=f"{p} writable")


def check_ports() -> CheckResult:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", DEFAULT_PORT))
        return CheckResult(
            name=f"port {DEFAULT_PORT}",
            status="ok",
            detail=f"port {DEFAULT_PORT} is free",
        )
    except OSError as e:
        return CheckResult(
            name=f"port {DEFAULT_PORT}",
            status="fail",
            detail=f"port {DEFAULT_PORT} bind failed: {e}",
            fix=f"identify the process with `ss -lntp | grep :{DEFAULT_PORT}` and stop it",
        )
    finally:
        s.close()


def check_docker() -> CheckResult:
    try:
        client = _docker_from_env()
        client.ping()
    except Exception as e:
        return CheckResult(
            name="docker",
            status="fail",
            detail=f"docker daemon unreachable: {e}",
            fix="ensure Docker is running and your user is in the 'docker' group",
        )
    try:
        info = client.info()
        version = client.version().get("Version", "?")
        runtimes = info.get("Runtimes", {})
        if "nvidia" not in runtimes:
            return CheckResult(
                name="docker",
                status="warn",
                detail=f"docker {version} OK, but nvidia runtime missing",
                fix="install nvidia-container-toolkit and restart dockerd",
            )
        return CheckResult(
            name="docker",
            status="ok",
            detail=f"docker {version} with nvidia runtime",
        )
    except Exception as e:
        return CheckResult(name="docker", status="warn", detail=f"docker reachable but info failed: {e}")


def check_gpus() -> CheckResult:
    if pynvml is None:
        return CheckResult(
            name="gpus",
            status="fail",
            detail="pynvml not available; cannot enumerate GPUs",
            fix="pip install pynvml",
        )
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
    except Exception as e:
        return CheckResult(
            name="gpus",
            status="fail",
            detail=f"NVML init failed: {e}",
            fix="install NVIDIA driver matching your CUDA runtime",
        )
    if n == 0:
        return CheckResult(name="gpus", status="fail", detail="no GPUs detected")
    names: list[str] = []
    total_mb = 0
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        nm = pynvml.nvmlDeviceGetName(h)
        nm = nm.decode() if isinstance(nm, bytes) else str(nm)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        names.append(nm)
        total_mb += int(mem.total) // 1024 // 1024
    summary = f"{n} GPU(s): {', '.join(names)} (total {total_mb} MB)"
    return CheckResult(name="gpus", status="ok", detail=summary)


def check_hf_token() -> CheckResult:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return CheckResult(name="HF token", status="ok", detail="HF_TOKEN set")
    return CheckResult(
        name="HF token",
        status="warn",
        detail="HF_TOKEN not set; gated models will fail to download",
        fix="export HF_TOKEN=hf_xxx (https://huggingface.co/settings/tokens)",
    )


def check_engine_images() -> CheckResult:
    """Check whether vLLM and SGLang images are cached locally."""
    try:
        client = _docker_from_env()
        tags = set()
        for img in client.images.list():
            tags.update(img.tags or [])
    except Exception:
        return CheckResult(
            name="engine images",
            status="warn",
            detail="docker not reachable; cannot inspect images",
        )
    found = []
    missing = []
    for prefix in ("vllm/vllm-openai:", "lmsysorg/sglang:"):
        hit = [t for t in tags if t.startswith(prefix)]
        if hit:
            found.append(hit[0])
        else:
            missing.append(prefix.rstrip(":"))
    if missing:
        return CheckResult(
            name="engine images",
            status="warn",
            detail=f"cached: {found or 'none'}; missing: {missing}",
            fix="serve will pull on first use; or `docker pull <image>` ahead of time",
        )
    return CheckResult(
        name="engine images",
        status="ok",
        detail=f"cached: {found}",
    )
```

- [ ] **Run + commit**

```bash
source .venv/bin/activate
pytest tests/unit/test_doctor.py -v
ruff check src/ tests/
git add src/serve_engine/doctor/ tests/unit/test_doctor.py
git commit -m "feat(doctor): per-check primitives (paths, ports, docker, gpus, HF, images)"
```

---

## Task 2: Doctor runner + `serve doctor` CLI

**Files:** `src/serve_engine/doctor/runner.py`, `src/serve_engine/cli/doctor_cmd.py`, `src/serve_engine/cli/__init__.py` (register)

- [ ] **`src/serve_engine/doctor/runner.py`**

```python
from __future__ import annotations

from serve_engine.doctor.checks import (
    CheckResult,
    check_docker,
    check_engine_images,
    check_gpus,
    check_hf_token,
    check_paths,
    check_ports,
)


def run_all() -> list[CheckResult]:
    return [
        check_paths(),
        check_ports(),
        check_docker(),
        check_gpus(),
        check_hf_token(),
        check_engine_images(),
    ]


def summarise(results: list[CheckResult]) -> tuple[int, int, int]:
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    fail = sum(1 for r in results if r.status == "fail")
    return ok, warn, fail
```

- [ ] **`src/serve_engine/cli/doctor_cmd.py`**

```python
from __future__ import annotations

import json

import typer

from serve_engine.cli import app
from serve_engine.doctor.runner import run_all, summarise

_GLYPH = {"ok": "✓", "warn": "!", "fail": "✗"}
_COLOR = {"ok": typer.colors.GREEN, "warn": typer.colors.YELLOW, "fail": typer.colors.RED}


@app.command("doctor")
def doctor(json_out: bool = typer.Option(False, "--json")):
    """Diagnose the local environment (Docker, GPUs, paths, ports, images)."""
    results = run_all()
    if json_out:
        typer.echo(json.dumps([{
            "name": r.name, "status": r.status, "detail": r.detail, "fix": r.fix
        } for r in results], indent=2))
        raise typer.Exit(_exit_code(results))
    for r in results:
        glyph = _GLYPH.get(r.status, "?")
        color = _COLOR.get(r.status, typer.colors.WHITE)
        typer.secho(f"  {glyph}  {r.name:<20} {r.detail}", fg=color)
        if r.fix and r.status != "ok":
            typer.echo(f"     → {r.fix}")
    ok, warn, fail = summarise(results)
    typer.echo()
    typer.secho(
        f"{ok} ok, {warn} warn, {fail} fail",
        fg=(typer.colors.RED if fail else (typer.colors.YELLOW if warn else typer.colors.GREEN)),
    )
    raise typer.Exit(_exit_code(results))


def _exit_code(results) -> int:
    if any(r.status == "fail" for r in results):
        return 1
    return 0
```

- [ ] **Register in `cli/__init__.py`** — add `doctor_cmd` to the import block in alphabetical position (between `daemon_cmd` and `key_cmd`).

- [ ] **Commit**

```bash
python -c "from serve_engine.cli import doctor_cmd; print('ok')"
pytest -v
ruff check src/ tests/
git add src/serve_engine/doctor/runner.py src/serve_engine/cli/doctor_cmd.py src/serve_engine/cli/__init__.py
git commit -m "feat(cli): serve doctor — environment diagnostic"
```

---

## Task 3: `serve setup` interactive wizard

**Files:** `src/serve_engine/cli/setup_cmd.py`, `src/serve_engine/cli/__init__.py` (register)

- [ ] **`src/serve_engine/cli/setup_cmd.py`**

```python
from __future__ import annotations

import asyncio
import time

import typer

from serve_engine import config
from serve_engine.cli import app, ipc
from serve_engine.doctor.runner import run_all, summarise


@app.command("setup")
def setup():
    """First-run wizard: doctor, start daemon, create admin key, print URL."""
    typer.echo("=== serve-engine setup ===")
    typer.echo()
    typer.echo("Step 1: environment diagnostic")
    results = run_all()
    _, _, fail = summarise(results)
    for r in results:
        glyph = {"ok": "✓", "warn": "!", "fail": "✗"}[r.status]
        typer.echo(f"  {glyph} {r.name}: {r.detail}")
    if fail:
        typer.secho(
            "\n✗ doctor reports failures; fix and re-run `serve setup`.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)

    typer.echo()
    typer.echo("Step 2: starting daemon")
    try:
        asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
        typer.echo("  daemon already running")
    except Exception:
        # subprocess-spawn via existing daemon_cmd path
        import subprocess
        import sys
        log_path = config.LOGS_DIR / "daemon.log"
        config.SERVE_DIR.mkdir(parents=True, exist_ok=True)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "serve_engine.daemon"],
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (config.SERVE_DIR / "daemon.pid").write_text(str(proc.pid))
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
                typer.echo(f"  daemon started (pid {proc.pid})")
                break
            except Exception:
                time.sleep(0.3)
        else:
            typer.secho("  daemon failed to come up; check logs", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)

    typer.echo()
    typer.echo("Step 3: create admin key")
    label = typer.prompt("Key label", default="admin")
    body = {"name": label, "tier": "admin"}
    result = asyncio.run(ipc.post(config.SOCK_PATH, "/admin/keys", json=body))
    typer.echo(f"  id:     {result['id']}")
    typer.echo(f"  secret: {result['secret']}")
    typer.echo()
    typer.echo("Save this secret — it won't be shown again.")
    typer.echo()
    typer.secho(
        f"Done. Open http://127.0.0.1:{config.DEFAULT_PUBLIC_PORT}/ and paste the secret.",
        fg=typer.colors.GREEN,
    )
```

- [ ] **Register in `cli/__init__.py`** (alphabetical position).

- [ ] **Commit**

```bash
python -c "from serve_engine.cli import setup_cmd; print('ok')"
pytest -v
ruff check src/ tests/
git add src/serve_engine/cli/setup_cmd.py src/serve_engine/cli/__init__.py
git commit -m "feat(cli): serve setup — first-run wizard"
```

---

## Task 4: `install.sh`

**Files:** `scripts/install.sh`

- [ ] **Write `scripts/install.sh`**

```bash
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
```

- [ ] **Make executable + commit**

```bash
chmod +x scripts/install.sh
bash -n scripts/install.sh  # syntax check
git add scripts/install.sh
git commit -m "feat(install): one-shot install.sh (uv tool install + doctor)"
```

---

## Task 5: Daemon-as-container

**Files:** `docker/daemon.Dockerfile`, `docker/README.md`

- [ ] **`docker/daemon.Dockerfile`**

```dockerfile
# serve-engine daemon as a container.
#
# Build:  docker build -f docker/daemon.Dockerfile -t serve-engine:dev .
# Run:    docker run -d --name serve --network host \
#             -v ~/.serve:/root/.serve \
#             -v /var/run/docker.sock:/var/run/docker.sock \
#             serve-engine:dev
#
# Note: --network host is the simplest way to let the daemon spawn sibling
# engine containers and reach them by 127.0.0.1:<host_port>. With a bridge
# network, the daemon container would need the engine containers on the
# same custom network and would address them by container name.

FROM python:3.12-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /opt/serve-engine
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

RUN uv pip install --system --no-cache .

EXPOSE 11500
VOLUME ["/root/.serve"]

CMD ["python3", "-m", "serve_engine.daemon", "--host", "0.0.0.0", "--port", "11500"]
```

- [ ] **`docker/README.md`**

```markdown
# Daemon-as-container

This directory contains a Dockerfile to run the serve-engine daemon itself in a container. The daemon spawns engine containers (vLLM, SGLang) on the host's Docker — so the container must have access to the host Docker socket.

## Build

```bash
docker build -f docker/daemon.Dockerfile -t serve-engine:dev .
```

## Run

```bash
docker run -d --name serve \
    --network host \
    -v ~/.serve:/root/.serve \
    -v /var/run/docker.sock:/var/run/docker.sock \
    serve-engine:dev
```

**Why `--network host`?** The daemon binds to `127.0.0.1:<allocated>` ports for the engine containers it spawns. On the host network, the daemon container resolves those addresses transparently. On a bridge network, it would need its own engine network and address-by-name routing.

**Security note:** Mounting `/var/run/docker.sock` grants root-equivalent privileges to anything inside the container. Don't combine this with untrusted code.

## Versioned tags

The pinned engine images in `backends/backends.yaml` are pulled lazily on first use. To pre-pull them:

```bash
docker exec serve serve pull-engine vllm
docker exec serve serve pull-engine sglang
```
```

- [ ] **Commit**

```bash
git add docker/daemon.Dockerfile docker/README.md
git commit -m "feat(docker): daemon-as-container Dockerfile + README"
```

---

## Task 6: Live verification

- [ ] **Run `serve doctor` against the real host**

```bash
source .venv/bin/activate
serve doctor
```

Expect a mix of ok/warn for any missing engine images. Exit code 0 if no failures.

- [ ] **Run `serve setup` (interactive — manual test)**

```bash
serve daemon stop 2>/dev/null
rm -f ~/.serve/db.sqlite*  # fresh state
serve setup
# Accept the default label "admin"
# Verify it prints a `sk-...` secret and the URL
```

- [ ] **Verify `install.sh` syntax**

```bash
bash -n scripts/install.sh
```

(Full end-to-end of `install.sh` requires a clean machine — not run here.)

## Verification

1. `pytest -v` — all tests pass.
2. `ruff check src/ tests/` clean.
3. `serve doctor` exits 0 on the H100 host.
4. `serve setup` walks through doctor, daemon, key creation, prints URL.

## Self-review

- **No external HTTP calls** in doctor (all checks are local). Lazy `pynvml.nvmlInit` only happens in `check_gpus`.
- **Daemon-as-container** uses `--network host` to keep the existing 127.0.0.1 routing intact. A bridge-network variant would require a re-architecture of `DockerClient.run`.
- **Placeholder scan:** none.
- **What's not covered yet** (future work): `serve update-engines` (image bump), `serve install-service` (systemd unit), Helm/k8s manifests.
