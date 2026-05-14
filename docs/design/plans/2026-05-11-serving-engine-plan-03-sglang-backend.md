# Serving Engine - Plan 03: SGLang Backend

**Goal:** Add SGLang as a second engine backend. Validate the `Backend` Protocol we built in Plan 01 actually pluggable. Engine selection is data-driven via YAML - autotune logic is parked.

**Architecture:** New `SGLangBackend` class implementing the same `Backend` Protocol as `VLLMBackend`. A `backends.yaml` manifest pins upstream image tags. A `selection.yaml` maps model patterns -> preferred engine. The manager picks the backend either from `--engine` CLI flag, the selection rules, or default vLLM.

**Tech Stack:** Same as Plans 01/02. New upstream image: `lmsysorg/sglang`. Note SGLang's launch command differs from vLLM - different argv shape - so this exercises the abstraction.

---

## File structure

```
serving-engine/
|-- src/serve_engine/
|   |-- backends/
|   |   |-- backends.yaml         # NEW - pinned image manifest
|   |   |-- selection.yaml        # NEW - model-pattern -> preferred backend
|   |   |-- selection.py          # NEW - pure-Python pattern matcher
|   |   +-- sglang.py             # NEW - SGLangBackend
|   |-- lifecycle/
|   |   +-- manager.py            # MODIFIED - uses Backend.from_manifest()
|   |-- daemon/
|   |   |-- __main__.py           # MODIFIED - load both backends
|   |   |-- admin.py              # MODIFIED - accept engine + selection override
|   |   +-- app.py                # MODIFIED - backends dict gets both
|   +-- cli/
|       +-- run_cmd.py            # MODIFIED - --engine flag
+-- tests/
    +-- unit/
        |-- test_sglang_backend.py     # NEW
        +-- test_selection.py          # NEW
```

---

## Task 1: Backend image manifest

**Files:**
- Create: `src/serve_engine/backends/backends.yaml`

The manifest is the single source of truth for default image tags. It is loaded by `Backend` classes at startup and overridable via config.

- [ ] **Step 1: Write `src/serve_engine/backends/backends.yaml`**

```yaml
# Pinned engine images. Update via `serve update-engines` (Plan 08) or
# `serve config set engine.<name>.image <tag>`.

vllm:
  image: vllm/vllm-openai
  pinned_tag: v0.20.2
  health_path: /health
  openai_base: /v1
  metrics_path: /metrics
  internal_port: 8000

sglang:
  image: lmsysorg/sglang
  pinned_tag: v0.5.5.post1
  health_path: /health
  openai_base: /v1
  metrics_path: /metrics
  internal_port: 30000
```

(SGLang's default port is 30000, not 8000. The manifest captures this.)

- [ ] **Step 2: Commit**

```bash
git add src/serve_engine/backends/backends.yaml
git commit -m "feat(backends): YAML manifest pinning vLLM v0.20.2 + SGLang v0.5.5.post1"
```

(No tests yet - Task 2 introduces the consumer.)

---

## Task 2: SGLang Backend class

**Files:**
- Create: `src/serve_engine/backends/sglang.py`
- Create: `tests/unit/test_sglang_backend.py`

SGLang's launch command is `python -m sglang.launch_server` - different from vLLM's `vllm serve`. Their official image's ENTRYPOINT IS the launcher, so we just pass argv (same shape as vLLM).

Key argv differences from vLLM (researched against SGLang's `launch_server --help`):
- `--model-path` instead of `--model`
- `--tp` instead of `--tensor-parallel-size`
- `--mem-fraction-static` instead of `--gpu-memory-utilization`
- `--context-length` instead of `--max-model-len`
- `--host 0.0.0.0 --port 30000` same
- `--served-model-name` same
- `--disable-cuda-graph` is a useful escape-hatch flag (default: enabled)
- `--enable-radix-cache` (default: on; the prefix-caching feature)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_sglang_backend.py`:
```python
from serve_engine.backends.sglang import SGLangBackend
from serve_engine.lifecycle.plan import DeploymentPlan


def _plan(**overrides):
    base = dict(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="sglang",
        image_tag="lmsysorg/sglang:v0.5.5.post1",
        gpu_ids=[0],
        max_model_len=8192,
    )
    base.update(overrides)
    return DeploymentPlan(**base)


def test_build_argv_minimum():
    argv = SGLangBackend().build_argv(_plan(), local_model_path="/models/llama-1b")
    assert "--model-path" in argv
    assert argv[argv.index("--model-path") + 1] == "/models/llama-1b"
    assert "--tp" in argv
    assert argv[argv.index("--tp") + 1] == "1"
    assert "--context-length" in argv
    assert argv[argv.index("--context-length") + 1] == "8192"
    assert "--mem-fraction-static" in argv
    assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "30000"


def test_build_argv_tp_4():
    argv = SGLangBackend().build_argv(
        _plan(gpu_ids=[0, 1, 2, 3], tensor_parallel=4),
        local_model_path="/models/x",
    )
    assert argv[argv.index("--tp") + 1] == "4"


def test_container_kwargs_gpu_request():
    kw = SGLangBackend().container_kwargs(_plan(gpu_ids=[2, 3], tensor_parallel=2))
    assert kw["device_requests"][0]["device_ids"] == ["2", "3"]
    assert kw["ipc_mode"] == "host"
    assert kw["shm_size"] == "2g"


def test_default_image():
    assert SGLangBackend.image_default.startswith("lmsysorg/sglang:")


def test_internal_port():
    assert SGLangBackend.internal_port == 30000
```

Run `pytest tests/unit/test_sglang_backend.py -v` -> FAIL (module missing).

(Note: `DeploymentPlan.backend` is typed `Literal["vllm"]` from Plan 01 / 02. Task 3 in this plan widens `SUPPORTED_BACKENDS` to include `"sglang"`. Until that lands, the test above will fail at `DeploymentPlan(...)` construction. That's fine - write the tests for the eventual signature; they pass after Task 3.)

- [ ] **Step 2: Implement `src/serve_engine/backends/sglang.py`**

```python
from __future__ import annotations

from typing import ClassVar

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.lifecycle.plan import DeploymentPlan

INTERNAL_PORT = 30000


class SGLangBackend:
    name: ClassVar[str] = "sglang"
    image_default: ClassVar[str] = "lmsysorg/sglang:v0.5.5.post1"
    health_path: ClassVar[str] = "/health"
    openai_base: ClassVar[str] = "/v1"
    metrics_path: ClassVar[str] = "/metrics"
    internal_port: ClassVar[int] = INTERNAL_PORT

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        argv: list[str] = [
            "--model-path", local_model_path,
            "--tp", str(plan.tensor_parallel),
            "--context-length", str(plan.max_model_len),
            "--mem-fraction-static", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype if plan.dtype != "auto" else "auto",
            "--host", "0.0.0.0",
            "--port", str(INTERNAL_PORT),
            "--served-model-name", plan.model_name,
        ]
        # SGLang's RadixAttention is on by default; explicit flag opt-out only.
        # enable_prefix_caching on the plan means "we want it", which matches default.
        for k, v in plan.extra_args.items():
            argv.extend([k, v])
        return argv

    def container_env(self, plan: DeploymentPlan) -> dict[str, str]:
        return {}

    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]:
        return {
            "device_requests": [
                {
                    "Driver": "nvidia",
                    "device_ids": [str(g) for g in plan.gpu_ids],
                    "Capabilities": [["gpu"]],
                }
            ],
            "ipc_mode": "host",
            "shm_size": "2g",
            "ulimits": [Ulimit(name="memlock", soft=-1, hard=-1)],
        }
```

- [ ] **Step 3: Commit (tests still failing - Task 3 fixes plan validation)**

```bash
git add src/serve_engine/backends/sglang.py tests/unit/test_sglang_backend.py
git commit -m "feat(backends): SGLang Backend (argv + container kwargs)"
```

---

## Task 3: Widen `DeploymentPlan.backend` to include sglang + add `internal_port` to Backend Protocol

**Files:**
- Modify: `src/serve_engine/lifecycle/plan.py`
- Modify: `src/serve_engine/backends/base.py`
- Modify: `src/serve_engine/backends/vllm.py`

The `Backend` Protocol gains an `internal_port` class attribute. vLLM gets `internal_port: ClassVar[int] = 8000`. SGLang already has 30000.

- [ ] **Step 1: Modify `src/serve_engine/lifecycle/plan.py`**

Change:
```python
SUPPORTED_BACKENDS = ("vllm",)
```
to:
```python
SUPPORTED_BACKENDS = ("vllm", "sglang")
```

Change the field annotation on `DeploymentPlan`:
```python
backend: Literal["vllm"]
```
to:
```python
backend: Literal["vllm", "sglang"]
```

And update the comment in the error message in `__post_init__`:
```python
raise ValueError(
    f"backend {self.backend!r} not supported "
    f"(supported: {SUPPORTED_BACKENDS})"
)
```

- [ ] **Step 2: Modify `src/serve_engine/backends/base.py`**

Add `internal_port: int` to the Protocol:
```python
class Backend(Protocol):
    name: str
    image_default: str
    health_path: str
    openai_base: str
    metrics_path: str
    internal_port: int

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]: ...
    def container_env(self, plan: DeploymentPlan) -> dict[str, str]: ...
    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]: ...
```

- [ ] **Step 3: Modify `src/serve_engine/backends/vllm.py`**

Add the class attribute (vLLM uses 8000):
```python
class VLLMBackend:
    name: ClassVar[str] = "vllm"
    image_default: ClassVar[str] = "vllm/vllm-openai:v0.20.2"
    health_path: ClassVar[str] = "/health"
    openai_base: ClassVar[str] = "/v1"
    metrics_path: ClassVar[str] = "/metrics"
    internal_port: ClassVar[int] = 8000
```

- [ ] **Step 4: Use `backend.internal_port` in manager.py**

In `src/serve_engine/lifecycle/manager.py`, find the line:
```python
internal_port=8000,
```
inside the `self._docker.run(...)` call, and change it to:
```python
internal_port=backend.internal_port,
```

- [ ] **Step 5: Update `tests/unit/test_plan.py` `test_plan_backend_must_be_supported`**

The current test uses `backend="trt-llm"`. Keep it - sglang now passes validation, but trt-llm still fails. The test asserts on the error message; it should still pass since `trt-llm` isn't in `("vllm", "sglang")`.

- [ ] **Step 6: Run tests**

```bash
source .venv/bin/activate
pytest -v
ruff check src/ tests/
```
Expected: all Plan 01/02 tests + 5 SGLang backend tests pass. Total: previous + 5 = 68.

- [ ] **Step 7: Commit**

```bash
git add src/serve_engine/lifecycle/plan.py src/serve_engine/backends/base.py src/serve_engine/backends/vllm.py src/serve_engine/lifecycle/manager.py
git commit -m "feat(backends): widen plan.backend to sglang + Backend.internal_port"
```

---

## Task 4: Engine selection

**Files:**
- Create: `src/serve_engine/backends/selection.yaml`
- Create: `src/serve_engine/backends/selection.py`
- Create: `tests/unit/test_selection.py`

Picks the preferred backend for a model name. Static patterns, no machine learning. Order: explicit user flag -> first matching pattern -> default (vllm).

- [ ] **Step 1: Write `src/serve_engine/backends/selection.yaml`**

```yaml
# Model-pattern -> preferred backend. First match wins.
# Patterns are case-insensitive fnmatch globs.
#
# Defaults to vLLM unless a rule matches. Override per-request with `--engine`.

rules:
  # Models that benefit from RadixAttention (multi-turn / agentic / long
  # shared prefixes). SGLang's prefix caching is more aggressive than vLLM's.
  - pattern: "*deepseek-v3*"
    backend: sglang
  - pattern: "*deepseek-r1*"
    backend: sglang
  - pattern: "*qwen*-vl*"
    backend: sglang   # vision-language; SGLang handles these well

default: vllm
```

- [ ] **Step 2: Write tests**

`tests/unit/test_selection.py`:
```python
from pathlib import Path

import pytest

from serve_engine.backends.selection import (
    SelectionConfig,
    load_selection,
    pick_backend,
)


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "selection.yaml"
    p.write_text(content)
    return p


def test_default_picks_vllm(tmp_path):
    cfg = load_selection(_write(tmp_path, "rules: []\ndefault: vllm\n"))
    assert pick_backend(cfg, "Llama-3.1-70B-Instruct") == "vllm"


def test_pattern_match(tmp_path):
    cfg = load_selection(_write(
        tmp_path,
        "rules:\n"
        "  - pattern: '*deepseek-v3*'\n"
        "    backend: sglang\n"
        "default: vllm\n",
    ))
    assert pick_backend(cfg, "deepseek-v3-671b") == "sglang"
    assert pick_backend(cfg, "Llama-3.1-70B") == "vllm"


def test_first_match_wins(tmp_path):
    cfg = load_selection(_write(
        tmp_path,
        "rules:\n"
        "  - pattern: '*qwen*'\n"
        "    backend: sglang\n"
        "  - pattern: '*qwen*-vl*'\n"
        "    backend: sglang\n"
        "default: vllm\n",
    ))
    assert pick_backend(cfg, "qwen-2.5-vl-7b") == "sglang"


def test_case_insensitive(tmp_path):
    cfg = load_selection(_write(
        tmp_path,
        "rules:\n"
        "  - pattern: '*DEEPSEEK*'\n"
        "    backend: sglang\n"
        "default: vllm\n",
    ))
    assert pick_backend(cfg, "deepseek-v3") == "sglang"


def test_load_default_path_uses_package_resource():
    # The packaged selection.yaml has rules + default fields.
    cfg = load_selection()
    assert cfg.default in ("vllm", "sglang")
    assert isinstance(cfg.rules, list)
```

- [ ] **Step 3: Implement `src/serve_engine/backends/selection.py`**

```python
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SelectionRule:
    pattern: str
    backend: str


@dataclass(frozen=True)
class SelectionConfig:
    rules: list[SelectionRule]
    default: str


def load_selection(path: Path | None = None) -> SelectionConfig:
    """Load selection rules from a file path, or from the packaged default."""
    if path is None:
        path_obj = files("serve_engine.backends").joinpath("selection.yaml")
        text = path_obj.read_text()
    else:
        text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    rules = [
        SelectionRule(pattern=r["pattern"], backend=r["backend"])
        for r in data.get("rules", [])
    ]
    return SelectionConfig(rules=rules, default=data.get("default", "vllm"))


def pick_backend(cfg: SelectionConfig, model_name: str) -> str:
    name_lower = model_name.lower()
    for rule in cfg.rules:
        if fnmatch.fnmatch(name_lower, rule.pattern.lower()):
            return rule.backend
    return cfg.default
```

- [ ] **Step 4: Add `pyyaml` dependency**

In `pyproject.toml`, add to `[project] dependencies`:
```
"pyyaml>=6.0",
```

Run `uv pip install -e ".[dev]"`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_selection.py -v
```
Expected: 5 pass.

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/backends/selection.yaml src/serve_engine/backends/selection.py tests/unit/test_selection.py pyproject.toml
git commit -m "feat(backends): YAML-driven engine selection rules"
```

---

## Task 5: Manager uses selection

**Files:**
- Modify: `src/serve_engine/daemon/admin.py`
- Modify: `tests/unit/test_admin_endpoints.py`

The admin endpoint accepts an optional `engine` field; if not set, the engine is chosen by `pick_backend(model_name)`.

- [ ] **Step 1: Modify `CreateDeploymentRequest` in `admin.py`**

Already accepts `backend: str = "vllm"`. Change the default to `None` and resolve via selection:

```python
class CreateDeploymentRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    hf_repo: str
    revision: str = "main"
    backend: str | None = None   # default -> selection rules
    image_tag: str | None = None
    gpu_ids: list[int]
    tensor_parallel: int | None = None
    max_model_len: int = 8192
    dtype: str = "auto"
    pinned: bool = False
    idle_timeout_s: int | None = None
    target_concurrency: int = 8
```

And in `create_deployment`:
```python
    from serve_engine.backends.selection import load_selection, pick_backend
    backend_name = body.backend
    if backend_name is None:
        backend_name = pick_backend(load_selection(), body.model_name)
    if backend_name not in backends:
        raise HTTPException(400, f"backend {backend_name!r} not supported")
    backend = backends[backend_name]
    image_tag = body.image_tag or backend.image_default
    ...
    plan = DeploymentPlan(
        ...
        backend=backend_name,
        ...
    )
```

- [ ] **Step 2: Add a test**

Append to `tests/unit/test_admin_endpoints.py`:

```python
@pytest.mark.asyncio
async def test_create_deployment_default_backend_is_vllm(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post(
            "/admin/deployments",
            json={
                "model_name": "x",
                "hf_repo": "org/x",
                "image_tag": "img:v1",
                "gpu_ids": [0],
                "max_model_len": 4096,
                # no `backend` field - should default via selection
            },
        )
    assert r.status_code == 201
    body = r.json()
    assert body["backend"] == "vllm"
```

The test fixture in `test_admin_endpoints.py` constructs `backends={"vllm": VLLMBackend()}` - that's fine. For Plan 03 we will also pass `sglang` in real startup (Task 6), but the test doesn't require it.

- [ ] **Step 3: Run tests, ruff, commit**

```bash
pytest -v
ruff check src/ tests/
git add src/serve_engine/daemon/admin.py tests/unit/test_admin_endpoints.py
git commit -m "feat(daemon): default backend chosen by selection rules"
```

---

## Task 6: Wire SGLang into daemon startup

**Files:**
- Modify: `src/serve_engine/daemon/__main__.py`

- [ ] **Step 1: Modify imports and the `backends` dict**

At the top of the file, add:
```python
from serve_engine.backends.sglang import SGLangBackend
```

In the `build_apps(...)` call inside `serve()`, change `backends={"vllm": VLLMBackend()}` to:
```python
    backends = {
        "vllm": VLLMBackend(),
        "sglang": SGLangBackend(),
    }
    tcp_app, uds_app = build_apps(
        conn=conn,
        docker_client=docker_client,
        backends=backends,
        models_dir=config.MODELS_DIR,
        topology=topology,
    )
```

- [ ] **Step 2: Smoke check + commit**

```bash
python -c "from serve_engine.daemon.__main__ import main; print('import ok')"
pytest -v
git add src/serve_engine/daemon/__main__.py
git commit -m "feat(daemon): register SGLang backend at daemon startup"
```

---

## Task 7: CLI `--engine` flag on `serve run`

**Files:**
- Modify: `src/serve_engine/cli/run_cmd.py`

- [ ] **Step 1: Add the option**

In `def run(...)`, after the existing options, add:
```python
    engine: str = typer.Option(
        None, "--engine",
        help="Force a specific engine (vllm | sglang). Default: auto-select.",
    ),
```

In the body, after building `body`, add:
```python
    if engine is not None:
        body["backend"] = engine
```

- [ ] **Step 2: Smoke check + commit**

```bash
python -c "from serve_engine.cli.run_cmd import run; print('ok')"
pytest -v
git add src/serve_engine/cli/run_cmd.py
git commit -m "feat(cli): --engine flag overrides backend selection"
```

---

## Task 8: Smoke v3 - load same model on both engines

**Files:**
- Create: `scripts/smoke_p03_engines.sh`

This is an additional smoke (does not replace `smoke_e2e.sh`). It loads Qwen 0.5B on vLLM, calls it, stops it, loads it on SGLang, calls it, verifies both produced output.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Plan 03 smoke: same model, two engines back-to-back.
# Prereqs same as smoke_e2e.sh + lmsysorg/sglang image cached (or
# pulled on first use - that takes ~3-5 min).

cleanup() {
    serve stop 2>/dev/null || true
    serve daemon stop 2>/dev/null || true
}
trap cleanup EXIT

serve daemon start
serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b

for engine in vllm sglang; do
    echo "=== loading qwen-0_5b on $engine ==="
    serve run qwen-0_5b --gpu 0 --ctx 4096 --engine "$engine"
    sleep 2
    curl -sS "http://127.0.0.1:11500/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"qwen-0_5b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":8,\"stream\":false}" \
      | tee "/tmp/serve_p03_$engine.out"
    echo
    grep -q "OK" "/tmp/serve_p03_$engine.out" || { echo "FAIL: no OK from $engine"; exit 1; }
    serve stop
    sleep 2
done

echo "PASS"
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/smoke_p03_engines.sh
git add scripts/smoke_p03_engines.sh
git commit -m "test: Plan 03 smoke (same model, two engines)"
```

---

## Verification (end of Plan 03)

1. `pytest -v` - all tests pass.
2. `ruff check src/ tests/` - clean.
3. `bash scripts/smoke_p03_engines.sh` on H100 - exits 0 with PASS.

## Self-review

- **Spec coverage:** SGLang backend (T2), selection rules (T4), engine pluggability (T3 internal_port), CLI `--engine` (T7), smoke (T8). Backend manifest (T1) lays groundwork for Plan 08's `serve update-engines`.
- **No autotune logic** - selection is purely static patterns. Correct per the parked-autotune decision.
- **Placeholder scan:** none.
- **Forward compat:** `SGLangBackend.container_kwargs` matches `VLLMBackend`'s shape. Adding TRT-LLM in a future plan is now demonstrably one new file.
