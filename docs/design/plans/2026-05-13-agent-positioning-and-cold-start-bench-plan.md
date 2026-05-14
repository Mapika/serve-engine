# Agent positioning + cold-start benchmark - implementation plan

**Goal:** Ship the open-source launch artifacts approved in `docs/design/specs/2026-05-13-agent-positioning-and-cold-start-bench.md`: a snapshot benchmark harness, three realistic agent recipes under `examples/`, a rewritten README anchored on the benchmark numbers, and a `LAUNCH.md` draft.

**Architecture:** No new daemon subsystems. The benchmark talks to the running daemon via the existing admin HTTP API over UDS (same transport pattern as the CLI). The examples are pure OpenAI-SDK Python clients with shell setup scripts - they treat serve-engine as a black box. The README rewrite is editorial.

**Tech Stack:** Python 3.11+, `httpx` (UDS transport), `openai` SDK, `faiss-cpu` (for recipe 02), `pytest` for the bench's pure-logic unit tests, the existing FastAPI daemon for integration.

**Execution discipline:**
- TDD applies strictly to the bench's pure-logic functions (aggregation, table output, GPU fingerprint parsing). For glue code that drives a real daemon, integration testing via an in-process fake admin server.
- Recipes are proofs-by-demonstration - no automated tests. Each `client.py` ships with a checked-in `sample-output.txt` from a real run; the diff against expected output is the canary.
- One git commit per task. Each commit leaves the repo in a green state (existing 363 tests still pass, ruff still clean).

**Tracking:** Use `- [ ]` checkboxes to mark progress. Tasks 1-5 are sequential (benchmark must produce real numbers before recipes/README make sense). Tasks 6-9 (examples) can be parallelized after Task 5 lands. Tasks 10-11 (README, LAUNCH) come last.

---

## Task 1: Benchmark skeleton - CLI, output shapes, GPU fingerprint

**Files:**
- Create: `scripts/__init__.py` (empty - makes `scripts` an importable package so the unit tests can `from scripts.bench_snapshots import ...`)
- Create: `scripts/bench_snapshots.py`
- Create: `tests/unit/test_bench_snapshots.py`

### Step 1.0: Make `scripts/` importable

```bash
touch scripts/__init__.py
```

Without this, `from scripts.bench_snapshots import ...` in the unit test fails under pytest because `scripts/` is currently a directory of standalone executables, not a Python package.

- [ ] **Step 1.0: Created `scripts/__init__.py`.**

### Step 1.1: Write the failing tests for `gpu_fingerprint` and `aggregate`

```python
# tests/unit/test_bench_snapshots.py
from __future__ import annotations

import statistics
from unittest.mock import patch

import pytest

from scripts.bench_snapshots import (
    RunResult,
    aggregate,
    gpu_fingerprint,
    to_markdown,
)


def test_gpu_fingerprint_parses_nvidia_smi():
    fake_output = "NVIDIA H100 80GB HBM3, 550.54.15, 81559 MiB\n"
    with patch("scripts.bench_snapshots.subprocess.check_output", return_value=fake_output):
        fp = gpu_fingerprint()
    assert fp["name"] == "NVIDIA H100 80GB HBM3"
    assert fp["driver"] == "550.54.15"
    assert fp["memory_mib"] == 81559


def test_gpu_fingerprint_returns_unknown_when_nvidia_smi_missing():
    with patch("scripts.bench_snapshots.subprocess.check_output", side_effect=FileNotFoundError):
        fp = gpu_fingerprint()
    assert fp == {"name": "unknown", "driver": "unknown", "memory_mib": 0}


def test_aggregate_computes_median_min_max():
    runs = [
        RunResult(ready_s=10.0, ttft_s=1.0, total_s=11.0, ok=True, error=None),
        RunResult(ready_s=11.0, ttft_s=1.2, total_s=12.2, ok=True, error=None),
        RunResult(ready_s=12.0, ttft_s=1.4, total_s=13.4, ok=True, error=None),
    ]
    agg = aggregate(runs)
    assert agg["ready"]["median"] == 11.0
    assert agg["ready"]["min"] == 10.0
    assert agg["ready"]["max"] == 12.0
    assert agg["ttft"]["median"] == 1.2
    assert agg["total"]["median"] == 12.2
    assert agg["n_ok"] == 3
    assert agg["n_failed"] == 0


def test_aggregate_ignores_failed_runs():
    runs = [
        RunResult(ready_s=10.0, ttft_s=1.0, total_s=11.0, ok=True, error=None),
        RunResult(ready_s=0.0, ttft_s=0.0, total_s=0.0, ok=False, error="oom"),
    ]
    agg = aggregate(runs)
    assert agg["n_ok"] == 1
    assert agg["n_failed"] == 1
    assert agg["ready"]["median"] == 10.0


def test_to_markdown_produces_three_row_table():
    cold = {
        "ready": {"median": 47.2, "min": 45.0, "max": 50.0},
        "ttft": {"median": 12.4, "min": 11.0, "max": 14.0},
        "total": {"median": 59.6, "min": 56.0, "max": 64.0},
        "n_ok": 5, "n_failed": 0,
    }
    warm = {
        "ready": {"median": 8.1, "min": 7.5, "max": 9.2},
        "ttft": {"median": 0.34, "min": 0.30, "max": 0.41},
        "total": {"median": 8.4, "min": 7.8, "max": 9.6},
        "n_ok": 5, "n_failed": 0,
    }
    md = to_markdown(cold, warm)
    assert "Engine ready" in md
    assert "First TTFT" in md
    assert "Total wallclock" in md
    assert "5.8" in md or "5.83" in md  # 47.2 / 8.1 ~ 5.83
    assert "36" in md  # 12.4 / 0.34 ~ 36.5
```

- [ ] **Step 1.1: Save the test file above to `tests/unit/test_bench_snapshots.py`.**

### Step 1.2: Run the tests - expect them to fail

Run: `pytest tests/unit/test_bench_snapshots.py -v`
Expected: All tests fail with `ImportError: cannot import name 'RunResult' from 'scripts.bench_snapshots'` (or `ModuleNotFoundError`).

- [ ] **Step 1.2: Verify the failures.**

### Step 1.3: Implement the skeleton + the four tested functions

```python
# scripts/bench_snapshots.py
#!/usr/bin/env python3
"""Cold-start snapshot benchmark for serve-engine.

Measures the wallclock cost of bringing a model from "not loaded" to
"first inference token" - once with no snapshot (cold), once with a
snapshot present (warm). The cold run produces the snapshot the warm
run consumes, so a single benchmark execution generates both.

See docs/design/specs/2026-05-13-agent-positioning-and-cold-start-bench.md
for methodology rationale.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    ready_s: float
    ttft_s: float
    total_s: float
    ok: bool
    error: str | None


def gpu_fingerprint() -> dict[str, Any]:
    """Return a small dict identifying the GPU. Falls back to 'unknown'
    when nvidia-smi is unavailable so the bench still produces output
    on non-NVIDIA machines (useful for development)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"name": "unknown", "driver": "unknown", "memory_mib": 0}
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    return {
        "name": parts[0],
        "driver": parts[1],
        "memory_mib": int(parts[2]),
    }


def aggregate(runs: list[RunResult]) -> dict[str, Any]:
    """Median / min / max across successful runs. Failed runs reported
    separately so callers can tell if the median is from 5 or 2 samples."""
    ok = [r for r in runs if r.ok]
    failed = [r for r in runs if not r.ok]
    if not ok:
        return {
            "ready": {"median": 0.0, "min": 0.0, "max": 0.0},
            "ttft": {"median": 0.0, "min": 0.0, "max": 0.0},
            "total": {"median": 0.0, "min": 0.0, "max": 0.0},
            "n_ok": 0,
            "n_failed": len(failed),
        }
    def col(getter):
        vals = [getter(r) for r in ok]
        return {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    return {
        "ready": col(lambda r: r.ready_s),
        "ttft": col(lambda r: r.ttft_s),
        "total": col(lambda r: r.total_s),
        "n_ok": len(ok),
        "n_failed": len(failed),
    }


def to_markdown(cold: dict[str, Any], warm: dict[str, Any]) -> str:
    """Produce the README-pasteable markdown table from two aggregates."""
    def speedup(c: float, w: float) -> str:
        return f"{c / w:.1f}x" if w > 0 else "n/a"
    rows = [
        ("Engine ready", cold["ready"]["median"], warm["ready"]["median"]),
        ("First TTFT", cold["ttft"]["median"], warm["ttft"]["median"]),
        ("Total wallclock", cold["total"]["median"], warm["total"]["median"]),
    ]
    lines = [
        "| Phase                | Cold (no snapshot) | Warm (from snapshot) | Speedup |",
        "|----------------------|--------------------:|---------------------:|--------:|",
    ]
    for name, c, w in rows:
        lines.append(f"| {name:<20} | {c:>16.2f}s | {w:>19.2f}s | {speedup(c, w):>7} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("docs/bench/snapshot-cold-vs-warm.json"))
    p.add_argument("--sock", type=Path, default=Path.home() / ".serve" / "sock")
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args()

    # Round-trip orchestration lands in Task 2-3. Skeleton just prints args + fingerprint.
    print(f"Would run {args.runs} cold + {args.runs} warm against {args.model} on GPU {args.gpu_id}")
    print(f"GPU: {gpu_fingerprint()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 1.3: Save the file above to `scripts/bench_snapshots.py`.**

### Step 1.4: Run the tests - expect green

Run: `pytest tests/unit/test_bench_snapshots.py -v`
Expected: 5 passed.

- [ ] **Step 1.4: Verify all 5 tests pass.**

### Step 1.5: Smoke-test the CLI

Run: `python scripts/bench_snapshots.py --help`
Expected: usage message including `--model`, `--runs`, `--output`, `--sock`, `--gpu-id`.

Run: `python scripts/bench_snapshots.py --model test/m --runs 1`
Expected: Two lines - `Would run 1 cold + 1 warm against test/m on GPU 0` and `GPU: {...}`. Exit 0.

- [ ] **Step 1.5: Verify CLI smoke-runs.**

### Step 1.6: Commit

```bash
git add scripts/__init__.py scripts/bench_snapshots.py tests/unit/test_bench_snapshots.py
git commit -m "feat(bench): skeleton for snapshot cold-vs-warm benchmark

Pure-logic functions (gpu_fingerprint, aggregate, to_markdown) plus
argparse CLI. Daemon round-trip lands in next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 1.6: Commit.**

---

## Task 2: One end-to-end run against the daemon (UDS admin + chat)

**Files:**
- Modify: `scripts/bench_snapshots.py` (add `AdminClient`, `one_run`)
- Modify: `tests/unit/test_bench_snapshots.py` (add integration test against in-process fake)

### Step 2.1: Write the failing integration test against an in-process fake daemon

Add to `tests/unit/test_bench_snapshots.py`:

```python
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI


@asynccontextmanager
async def fake_daemon(tmp_path):
    """Spin up a FastAPI app on a UDS socket that simulates the admin
    surface bench_snapshots needs: POST /admin/models, POST /admin/deployments
    (returns id), GET /admin/deployments/{id} (status flips loading->ready
    after one poll), DELETE /admin/deployments/{id}, plus a fake chat
    completion endpoint that streams a single token."""
    import uvicorn
    from contextlib import suppress

    app = FastAPI()
    state = {"next_id": 1, "deps": {}, "poll_count": {}}

    @app.post("/admin/models", status_code=201)
    async def create_model(body: dict):
        return {"name": body["name"]}

    @app.post("/admin/deployments", status_code=201)
    async def create_dep(body: dict):
        did = state["next_id"]
        state["next_id"] += 1
        state["deps"][did] = {"id": did, "status": "loading", "model_name": body["model_name"]}
        state["poll_count"][did] = 0
        return state["deps"][did]

    @app.get("/admin/deployments")
    async def list_deps():
        out = []
        for did, d in state["deps"].items():
            state["poll_count"][did] += 1
            if state["poll_count"][did] >= 2 and d["status"] == "loading":
                d["status"] = "ready"
            out.append(d)
        return out

    @app.delete("/admin/deployments/{dep_id}", status_code=204)
    async def del_dep(dep_id: int):
        state["deps"].pop(dep_id, None)
        return None

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        from fastapi.responses import StreamingResponse
        async def gen():
            await asyncio.sleep(0.01)
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")

    sock = tmp_path / "fake.sock"
    config = uvicorn.Config(app, uds=str(sock), log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # wait for the socket to appear
    for _ in range(50):
        if sock.exists():
            break
        await asyncio.sleep(0.02)
    try:
        yield sock
    finally:
        server.should_exit = True
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_one_run_against_fake_daemon(tmp_path):
    from scripts.bench_snapshots import AdminClient, one_run
    async with fake_daemon(tmp_path) as sock:
        client = AdminClient(sock)
        result = await one_run(
            client,
            model_name="test-model",
            hf_repo="test/m",
            gpu_id=0,
            poll_interval_s=0.01,
        )
    assert result.ok is True
    assert result.error is None
    assert result.ready_s > 0
    assert result.ttft_s > 0
    assert result.total_s >= result.ready_s + result.ttft_s - 0.001  # rounding
```

- [ ] **Step 2.1: Add the test above.**

### Step 2.2: Run the test - expect failure

Run: `pytest tests/unit/test_bench_snapshots.py::test_one_run_against_fake_daemon -v`
Expected: ImportError for `AdminClient` and `one_run`.

- [ ] **Step 2.2: Verify the failure.**

### Step 2.3: Implement `AdminClient` and `one_run`

Add to `scripts/bench_snapshots.py` (above `def main()`):

```python
import time
import httpx


class AdminClient:
    """Thin async client over the daemon UDS socket. UDS path bypasses
    admin auth (the daemon trusts anything on the socket - same trust
    model the CLI uses)."""

    def __init__(self, sock: Path):
        self._sock = sock

    def _client(self) -> httpx.AsyncClient:
        transport = httpx.AsyncHTTPTransport(uds=str(self._sock))
        return httpx.AsyncClient(
            transport=transport,
            base_url="http://localhost",
            timeout=120.0,
        )

    async def ensure_model(self, name: str, hf_repo: str) -> None:
        """Register the model. If it already exists, the daemon returns
        4xx - that's fine, we swallow it."""
        async with self._client() as c:
            r = await c.post("/admin/models", json={"name": name, "hf_repo": hf_repo})
            if r.status_code not in (200, 201, 409):
                r.raise_for_status()

    async def create_deployment(self, model_name: str, hf_repo: str, gpu_id: int) -> int:
        async with self._client() as c:
            r = await c.post(
                "/admin/deployments",
                json={
                    "model_name": model_name,
                    "hf_repo": hf_repo,
                    "gpu_ids": [gpu_id],
                    "pinned": False,
                    "max_model_len": 4096,
                },
            )
            r.raise_for_status()
            return r.json()["id"]

    async def wait_ready(self, dep_id: int, poll_interval_s: float, timeout_s: float = 600.0) -> None:
        deadline = time.monotonic() + timeout_s
        async with self._client() as c:
            while True:
                r = await c.get("/admin/deployments")
                r.raise_for_status()
                deps = {d["id"]: d for d in r.json()}
                d = deps.get(dep_id)
                if d is None:
                    raise RuntimeError(f"deployment {dep_id} disappeared")
                if d["status"] == "ready":
                    return
                if d["status"] == "failed":
                    raise RuntimeError(f"deployment {dep_id} failed: {d.get('last_error')}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"deployment {dep_id} not ready after {timeout_s}s")
                await asyncio.sleep(poll_interval_s)

    async def first_ttft(self, model_name: str) -> float:
        """Send a chat completion, measure wallclock to first byte of the
        first SSE token chunk. Uses streaming so we genuinely measure TTFT,
        not full-response latency."""
        async with self._client() as c:
            t0 = time.monotonic()
            async with c.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 4,
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if b"data:" in chunk and b"[DONE]" not in chunk:
                        return time.monotonic() - t0
            raise RuntimeError("stream ended without a token")

    async def delete_deployment(self, dep_id: int) -> None:
        async with self._client() as c:
            await c.delete(f"/admin/deployments/{dep_id}")


async def one_run(
    client: AdminClient,
    *,
    model_name: str,
    hf_repo: str,
    gpu_id: int,
    poll_interval_s: float = 1.0,
) -> RunResult:
    """One full timed cycle: create deployment -> wait ready -> measure TTFT -> delete."""
    try:
        await client.ensure_model(model_name, hf_repo)
        t0 = time.monotonic()
        dep_id = await client.create_deployment(model_name, hf_repo, gpu_id)
        await client.wait_ready(dep_id, poll_interval_s)
        ready_s = time.monotonic() - t0
        ttft_s = await client.first_ttft(model_name)
        await client.delete_deployment(dep_id)
        return RunResult(
            ready_s=ready_s,
            ttft_s=ttft_s,
            total_s=ready_s + ttft_s,
            ok=True,
            error=None,
        )
    except Exception as e:
        return RunResult(ready_s=0.0, ttft_s=0.0, total_s=0.0, ok=False, error=str(e))
```

- [ ] **Step 2.3: Add the code above.**

### Step 2.4: Run the test - expect green

Run: `pytest tests/unit/test_bench_snapshots.py -v`
Expected: 6 passed (5 prior + 1 new).

- [ ] **Step 2.4: Verify all tests pass.**

### Step 2.5: Commit

```bash
git add scripts/bench_snapshots.py tests/unit/test_bench_snapshots.py
git commit -m "feat(bench): admin HTTP client + one-run orchestration

AdminClient wraps the UDS transport (same pattern as cli/ipc.py).
one_run() drives a single create->ready->TTFT->delete cycle and returns
a RunResult dataclass. Tested against an in-process FastAPI fake.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2.5: Commit.**

---

## Task 3: Cold/warm orchestration + snapshot dir handling

**Files:**
- Modify: `scripts/bench_snapshots.py` (add `benchmark()` + snapshot helpers)
- Modify: `tests/unit/test_bench_snapshots.py` (add orchestration test)

### Step 3.1: Write the failing test for `benchmark()`

Add to `tests/unit/test_bench_snapshots.py`:

```python
@pytest.mark.asyncio
async def test_benchmark_runs_cold_then_warm(tmp_path, monkeypatch):
    """Cold runs delete the snapshot dir first; warm runs do not.
    benchmark() returns two lists of RunResult."""
    from scripts.bench_snapshots import benchmark

    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    fake_snapshot = snapshots_dir / "abc123"
    fake_snapshot.mkdir()
    (fake_snapshot / "marker").write_text("present")

    async with fake_daemon(tmp_path) as sock:
        result = await benchmark(
            sock=sock,
            model_name="test-model",
            hf_repo="test/m",
            gpu_id=0,
            runs=2,
            snapshots_dir=snapshots_dir,
            poll_interval_s=0.01,
        )
    assert len(result["cold"]) == 2
    assert len(result["warm"]) == 2
    # Cold phase started by deleting the dir; warm phase ran with it present.
    # Hard to assert from outside without instrumentation - accept that the
    # call sequence ran without raising.
    assert all(r.ok for r in result["cold"])
    assert all(r.ok for r in result["warm"])
```

- [ ] **Step 3.1: Add the test.**

### Step 3.2: Run - expect failure

Run: `pytest tests/unit/test_bench_snapshots.py::test_benchmark_runs_cold_then_warm -v`
Expected: ImportError on `benchmark`.

- [ ] **Step 3.2: Verify failure.**

### Step 3.3: Implement `benchmark()` and snapshot helpers

Add to `scripts/bench_snapshots.py` (above `def main()`):

```python
import shutil


def _purge_snapshots(snapshots_dir: Path) -> None:
    """Delete all snapshot directories. Used before each cold run.
    We delete the whole dir contents rather than guessing snapshot_keys
    because the key is content-addressable on the deployment config  -
    simpler and equivalent for the bench's purposes."""
    if not snapshots_dir.is_dir():
        return
    for child in snapshots_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


async def _wait_for_snapshot_save(snapshots_dir: Path, timeout_s: float = 120.0) -> bool:
    """Snapshot save is async (manager schedules it post-deployment).
    Poll for any non-empty subdir under snapshots_dir. Returns True if
    a snapshot appears within timeout_s, False otherwise (failure is
    not fatal; warm runs will then be effectively cold)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if snapshots_dir.is_dir():
            for child in snapshots_dir.iterdir():
                if child.is_dir() and any(child.iterdir()):
                    return True
        await asyncio.sleep(1.0)
    return False


async def benchmark(
    *,
    sock: Path,
    model_name: str,
    hf_repo: str,
    gpu_id: int,
    runs: int,
    snapshots_dir: Path,
    poll_interval_s: float = 1.0,
) -> dict[str, list[RunResult]]:
    """Run `runs` cold passes then `runs` warm passes. Returns both lists."""
    client = AdminClient(sock)

    cold: list[RunResult] = []
    for i in range(runs):
        _purge_snapshots(snapshots_dir)
        r = await one_run(
            client,
            model_name=model_name,
            hf_repo=hf_repo,
            gpu_id=gpu_id,
            poll_interval_s=poll_interval_s,
        )
        cold.append(r)
        # Give the daemon time to persist the snapshot before the next cold run
        # purges it. (Last cold run's snapshot is what warm runs consume.)
        if r.ok and i == runs - 1:
            await _wait_for_snapshot_save(snapshots_dir)

    warm: list[RunResult] = []
    for _ in range(runs):
        r = await one_run(
            client,
            model_name=model_name,
            hf_repo=hf_repo,
            gpu_id=gpu_id,
            poll_interval_s=poll_interval_s,
        )
        warm.append(r)

    return {"cold": cold, "warm": warm}
```

- [ ] **Step 3.3: Add the code.**

### Step 3.4: Run - expect green

Run: `pytest tests/unit/test_bench_snapshots.py -v`
Expected: 7 passed.

- [ ] **Step 3.4: Verify all tests pass.**

### Step 3.5: Commit

```bash
git add scripts/bench_snapshots.py tests/unit/test_bench_snapshots.py
git commit -m "feat(bench): cold/warm orchestration + snapshot dir handling

benchmark() runs N cold passes (snapshot dir purged before each), waits
for the daemon's async snapshot save to settle after the last cold pass,
then N warm passes. Both phases share the same admin client.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3.5: Commit.**

---

## Task 4: Wire `main()` - JSON output, markdown table, exit codes

**Files:**
- Modify: `scripts/bench_snapshots.py` (replace stub `main` body)

### Step 4.1: Replace `main()` with the real implementation

Replace the existing `main()` in `scripts/bench_snapshots.py` with:

```python
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF repo id; also used as the registered model name (with / -> -).")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("docs/bench/snapshot-cold-vs-warm.json"))
    p.add_argument("--sock", type=Path, default=Path.home() / ".serve" / "sock")
    p.add_argument("--snapshots-dir", type=Path,
                   default=Path.home() / ".serve" / "snapshots")
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args()

    if not args.sock.exists():
        print(f"error: daemon socket not found at {args.sock} - is the daemon running?",
              file=sys.stderr)
        return 2

    model_name = args.model.split("/")[-1].lower()
    hf_repo = args.model

    print(f"Running {args.runs} cold + {args.runs} warm passes for {hf_repo}")
    print(f"GPU: {gpu_fingerprint()}")

    result = asyncio.run(benchmark(
        sock=args.sock,
        model_name=model_name,
        hf_repo=hf_repo,
        gpu_id=args.gpu_id,
        runs=args.runs,
        snapshots_dir=args.snapshots_dir,
    ))

    cold_agg = aggregate(result["cold"])
    warm_agg = aggregate(result["warm"])

    table = to_markdown(cold_agg, warm_agg)
    print("\n" + table + "\n")

    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    payload = {
        "model": hf_repo,
        "engine": "vllm",
        "runs": args.runs,
        "hardware": gpu_fingerprint(),
        "serve_engine_git_sha": git_sha,
        "cold": {"runs": [asdict(r) for r in result["cold"]], "aggregate": cold_agg},
        "warm": {"runs": [asdict(r) for r in result["warm"]], "aggregate": warm_agg},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")

    # Non-zero exit if too many runs failed - caller can gate launch on this.
    if cold_agg["n_failed"] > args.runs // 2 or warm_agg["n_failed"] > args.runs // 2:
        print("FAILED: majority of runs errored", file=sys.stderr)
        return 1
    return 0
```

- [ ] **Step 4.1: Replace `main()` as shown.**

### Step 4.2: Smoke-test the CLI surface (no daemon needed)

Run: `python scripts/bench_snapshots.py --sock /nonexistent 2>&1; echo "exit=$?"`
Expected: `error: daemon socket not found at /nonexistent` and `exit=2`.

- [ ] **Step 4.2: Verify the missing-socket exit path.**

### Step 4.3: Run unit tests one more time to confirm nothing broke

Run: `pytest tests/unit/test_bench_snapshots.py -v && ruff check scripts/ src/ tests/`
Expected: 7 passed, ruff clean.

- [ ] **Step 4.3: Verify.**

### Step 4.4: Commit

```bash
git add scripts/bench_snapshots.py
git commit -m "feat(bench): main() - JSON output, markdown table, exit codes

Non-zero exit if a majority of runs fail so a CI / launch script can
gate on the result. JSON payload includes hardware fingerprint + git
SHA for reproducibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4.4: Commit.**

---

## Task 5: Run the benchmark against a real engine - the gate

**This is the critical-path checkpoint per the spec.** If the First-TTFT row doesn't show >=5x speedup, **stop and reassess before any further work**. The spec authorizes that pause.

**Files:**
- Create: `docs/bench/snapshot-cold-vs-warm.json`

### Step 5.1: Pre-flight

Run these in separate terminals or sequentially:

```bash
# 1. Ensure daemon is running
serve daemon start

# 2. Pre-pull weights so HF download isn't in the timing
serve pull Qwen/Qwen2.5-1.5B-Instruct --name qwen2.5-1.5b-instruct

# 3. Verify the model is registered + weights are on disk
serve ls
ls ~/.serve/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/
```

Expected: daemon running, `serve ls` shows `qwen2.5-1.5b-instruct`, model weights present.

- [ ] **Step 5.1: Pre-flight done.**

### Step 5.2: Run the benchmark

```bash
python scripts/bench_snapshots.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --runs 5 \
    --output docs/bench/snapshot-cold-vs-warm.json
```

Expected: ~20-30 minutes wall clock. Final stdout includes the markdown table.

- [ ] **Step 5.2: Bench runs to completion, JSON written.**

### Step 5.3: Inspect numbers + check the gate

```bash
cat docs/bench/snapshot-cold-vs-warm.json | python -c "
import json, sys
d = json.load(sys.stdin)
cold_ttft = d['cold']['aggregate']['ttft']['median']
warm_ttft = d['warm']['aggregate']['ttft']['median']
ratio = cold_ttft / warm_ttft if warm_ttft > 0 else 0
print(f'cold TTFT median = {cold_ttft:.2f}s')
print(f'warm TTFT median = {warm_ttft:.2f}s')
print(f'speedup = {ratio:.1f}x')
print('GATE PASS' if ratio >= 5.0 else 'GATE FAIL - STOP AND REASSESS')
"
```

- [ ] **Step 5.3: Gate check.** If output ends `GATE PASS`, continue. If `GATE FAIL`, do not proceed - return to brainstorming and revise the spec's launch story.

### Step 5.4: Commit the JSON (only if gate passed)

```bash
git add docs/bench/snapshot-cold-vs-warm.json
git commit -m "bench(snapshot): record cold-vs-warm numbers for Qwen2.5-1.5B-Instruct

$(python -c "import json; d=json.load(open('docs/bench/snapshot-cold-vs-warm.json'));
print(f'  cold TTFT: {d[\"cold\"][\"aggregate\"][\"ttft\"][\"median\"]:.2f}s')")

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5.4: Commit (only if Step 5.3 passed).**

---

## Task 6: `examples/README.md` - recipe index

**Files:**
- Create: `examples/README.md`

### Step 6.1: Write the index

Create `examples/README.md`:

```markdown
# Examples

Realistic agent workloads that run end-to-end against a `serve-engine`
daemon. Each recipe is self-contained:

```
NN-<name>/
|-- README.md          # what this recipe shows, expected output
|-- setup.sh           # pulls models, starts deployments
|-- client.py          # OpenAI Python SDK only - no serve-engine imports
+-- sample-output.txt  # checked-in output from a real run, for verification
```

| Recipe | Demonstrates | VRAM | Models |
|---|---|---|---|
| [01-router-reasoner](01-router-reasoner/) | Multi-model auto-swap with a cost-saving routing pattern | ~6 GB | Qwen2.5-0.5B + Qwen2.5-1.5B |
| [02-rag-embed-chat](02-rag-embed-chat/) | Embeddings + chat from one daemon - RAG over serve-engine's own README | ~5 GB | bge-small-en-v1.5 + Qwen2.5-1.5B |
| [03-lora-per-task](03-lora-per-task/) | LoRA hot-load - switch task adapters without restarting the engine | ~6 GB | Qwen2.5-1.5B + 2 public LoRAs |

## Prerequisites

- `serve-engine` installed and a daemon running (`serve daemon start`).
- An admin API key (`serve key create demo --tier admin`) exported as `OPENAI_API_KEY`.
- `pip install openai` (recipe 02 also needs `faiss-cpu numpy`).

## Recommended reading order

If you're new: start with **01** - it's the cleanest demonstration of the
"many models, one endpoint" story. **02** shows heterogeneous model
families. **03** is the differentiator most other tools can't do at all.
```

- [ ] **Step 6.1: Create the file.**

### Step 6.2: Commit

```bash
mkdir -p examples
git add examples/README.md
git commit -m "docs(examples): top-level recipe index

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6.2: Commit.**

---

## Task 7: Recipe 01 - router-reasoner

**Files:**
- Create: `examples/01-router-reasoner/README.md`
- Create: `examples/01-router-reasoner/setup.sh`
- Create: `examples/01-router-reasoner/client.py`
- Create: `examples/01-router-reasoner/sample-output.txt` (after run)

### Step 7.1: Write `setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Recipe 01 - Router + reasoner
#
# Pulls a small router model (0.5B) and a medium reasoner (1.5B), then
# pins both. They live in serve-engine as separate models; client.py
# chooses between them per request.

serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b
serve pull Qwen/Qwen2.5-1.5B-Instruct --name qwen-1_5b
serve run qwen-0_5b --gpu 0 --pin
serve run qwen-1_5b --gpu 0 --pin
serve ps
echo
echo "Both models loaded. Run: python client.py"
```

- [ ] **Step 7.1: Create + `chmod +x examples/01-router-reasoner/setup.sh`.**

### Step 7.2: Write `client.py`

```python
"""Recipe 01 - Router + reasoner.

Demonstrates multi-model serving on one daemon:
  - A small (0.5B) router model classifies each prompt as trivial or complex.
  - Trivial prompts are answered by the same small model.
  - Complex prompts are escalated to a medium (1.5B) reasoner.

The client talks to one OpenAI-compatible endpoint and picks the model
per request by setting `model=`. serve-engine handles concurrent serving
of both pinned models from one GPU.
"""
from __future__ import annotations

import os
import time

from openai import OpenAI

ROUTER_MODEL = "qwen-0_5b"
REASONER_MODEL = "qwen-1_5b"
BASE_URL = os.environ.get("SERVE_URL", "http://127.0.0.1:11500/v1")

PROMPTS = [
    "What is 2 + 2?",
    "Explain why the sky is blue in one sentence.",
    "List three primary colors.",
    "Prove that the square root of 2 is irrational.",
    "What is the capital of France?",
    "Derive the quadratic formula from first principles.",
    "Translate 'hello' to Spanish.",
    "Compare and contrast utilitarian and deontological ethics.",
    "What's the boiling point of water in Celsius?",
    "Explain how RSA encryption works.",
    "Who wrote Hamlet?",
    "Discuss the implications of Godel's incompleteness theorems.",
    "What's 5 times 7?",
    "Describe the architectural differences between transformers and RNNs.",
    "Is the Earth round or flat?",
    "Analyze the geopolitical consequences of the fall of the Berlin Wall.",
    "Name a planet in our solar system.",
    "Explain quantum entanglement in detail.",
    "What language is spoken in Japan?",
    "Construct a proof that there are infinitely many primes.",
]

ROUTER_SYSTEM = (
    "Classify the user's prompt as exactly one of: SIMPLE or COMPLEX. "
    "SIMPLE means a one-line factual answer suffices. COMPLEX means "
    "the question needs explanation, derivation, or analysis. "
    "Respond with only the single word SIMPLE or COMPLEX."
)


def classify(client: OpenAI, prompt: str) -> str:
    r = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4,
        temperature=0.0,
    )
    label = (r.choices[0].message.content or "").strip().upper()
    return "COMPLEX" if "COMPLEX" in label else "SIMPLE"


def answer(client: OpenAI, model: str, prompt: str) -> tuple[str, int]:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    content = r.choices[0].message.content or ""
    tokens = r.usage.completion_tokens if r.usage else len(content.split())
    return content.strip(), tokens


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "no-auth"))

    routed_simple = routed_complex = 0
    tokens_small = tokens_large = 0
    t0 = time.monotonic()

    for prompt in PROMPTS:
        label = classify(client, prompt)
        target = REASONER_MODEL if label == "COMPLEX" else ROUTER_MODEL
        text, n_tok = answer(client, target, prompt)
        if target == REASONER_MODEL:
            routed_complex += 1
            tokens_large += n_tok
        else:
            routed_simple += 1
            tokens_small += n_tok
        print(f"[{label:7}] {prompt[:60]:<60} -> {text[:80]}")

    wall = time.monotonic() - t0
    print()
    print(f"Routed {routed_simple} to {ROUTER_MODEL} ({tokens_small} tok), "
          f"{routed_complex} to {REASONER_MODEL} ({tokens_large} tok)")
    print(f"Wall: {wall:.1f}s")
    # Estimate: 1.5B costs roughly 3x compute per token vs 0.5B; if we
    # had sent everything to the reasoner the cost would have been
    # (tokens_small + tokens_large) at the larger-model rate.
    baseline = (tokens_small + tokens_large) * 3
    actual = tokens_small * 1 + tokens_large * 3
    savings_pct = 100 * (baseline - actual) / baseline if baseline else 0
    print(f"Est. compute saved vs. always-{REASONER_MODEL}: {savings_pct:.0f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Create `examples/01-router-reasoner/client.py`.**

### Step 7.3: Write `README.md`

```markdown
# 01 - Router + reasoner

**What this shows:** Two models served from one daemon, with the client
choosing per request. Small model handles trivial questions cheaply;
big model handles the hard ones.

## Setup

```bash
./setup.sh                    # pulls + pins both models
serve key create demo --tier admin
export OPENAI_API_KEY=sk-...  # the secret from the previous command
pip install openai
```

## Run

```bash
python client.py
```

## What to expect

20 mixed prompts. Each is first classified by the 0.5B router as
SIMPLE or COMPLEX, then answered by the appropriate model. Final output
includes a tokens-routed summary and an estimated compute savings vs.
always-using-the-big-model.

See `sample-output.txt` for a real recorded run.

## Why this matters

A typical agent workload is dominated by routine routing decisions
that don't need a 70B model. With serve-engine you keep multiple
models pinned on one GPU and pick per-request - no inference proxy,
no separate processes, one OpenAI endpoint.
```

- [ ] **Step 7.3: Create the file.**

### Step 7.4: Run the recipe end-to-end and capture sample output

```bash
cd examples/01-router-reasoner
./setup.sh
python client.py | tee sample-output.txt
cd ../..
```

Expected: 20 lines of classification + answer, then a summary. Watch for any "model not found" errors - if so, the model name doesn't match `setup.sh`. Fix and re-run.

- [ ] **Step 7.4: Run the recipe; `sample-output.txt` produced.**

### Step 7.5: Commit

```bash
git add examples/01-router-reasoner/
git commit -m "feat(examples): recipe 01 - router-reasoner

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7.5: Commit.**

---

## Task 8: Recipe 02 - RAG embed + chat

**Pre-task validation:** Confirm that `serve-engine`'s OpenAI proxy supports `/v1/embeddings` correctly for `BAAI/bge-small-en-v1.5`. The spec flagged this as an open implementation question. **If embeddings don't work, swap in a different recipe (e.g., "two chat models for verifier-pattern") - do not silently ship a broken example.**

**Files:**
- Create: `examples/02-rag-embed-chat/README.md`
- Create: `examples/02-rag-embed-chat/setup.sh`
- Create: `examples/02-rag-embed-chat/docs/` (5-10 text files extracted from project README)
- Create: `examples/02-rag-embed-chat/client.py`
- Create: `examples/02-rag-embed-chat/sample-output.txt` (after run)

### Step 8.1: Pre-task - validate embeddings work

```bash
serve pull BAAI/bge-small-en-v1.5 --name bge-small
serve run bge-small --gpu 0 --pin
curl http://127.0.0.1:11500/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-small","input":"hello world"}' | python -m json.tool
```

Expected: a JSON response with `data[0].embedding` being a list of ~384 floats.
If it 4xx/5xx, **stop** and either fix the embeddings proxy first or replace recipe 02 with a verifier-pattern recipe (e.g., 1.5B writes, 0.5B critiques, 1.5B revises).

- [ ] **Step 8.1: Embeddings validated against the daemon.**

### Step 8.2: Write `setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

serve pull BAAI/bge-small-en-v1.5 --name bge-small
serve pull Qwen/Qwen2.5-1.5B-Instruct --name qwen-1_5b
serve run bge-small --gpu 0 --pin
serve run qwen-1_5b --gpu 0 --pin
serve ps
echo
echo "Both models loaded. Run: python client.py"
```

- [ ] **Step 8.2: Create + chmod +x.**

### Step 8.3: Extract a small corpus

```bash
mkdir -p examples/02-rag-embed-chat/docs
```

Then split serve-engine's `README.md` into 8 short markdown files (~80-200 words each), one topic per file. Suggested split:
- `01-what-it-does.md`
- `02-requirements.md`
- `03-install.md`
- `04-quickstart.md`
- `05-cli.md`
- `06-architecture.md`
- `07-disk-layout.md`
- `08-performance.md`

Each file is a few paragraphs lifted from the corresponding README section.

- [ ] **Step 8.3: Create 8 short markdown files under `examples/02-rag-embed-chat/docs/`.**

### Step 8.4: Write `client.py`

```python
"""Recipe 02 - RAG with embeddings + chat from one daemon.

Embeds 8 short docs about serve-engine, builds an in-memory FAISS index,
then answers questions by retrieving the top-3 chunks and feeding them
to the chat model. All traffic goes to one daemon, one OpenAI endpoint;
the only difference between calls is `model=`.
"""
from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

EMBED_MODEL = "bge-small"
CHAT_MODEL = "qwen-1_5b"
BASE_URL = os.environ.get("SERVE_URL", "http://127.0.0.1:11500/v1")

DOCS_DIR = Path(__file__).parent / "docs"

QUESTIONS = [
    "What problem does serve-engine solve?",
    "What engines does it support?",
    "How do I install it?",
    "Where does the daemon store state on disk?",
]


def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "no-auth"))

    # Load corpus
    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    chunks = [(p.name, p.read_text()) for p in doc_paths]
    names, texts = zip(*chunks, strict=False)
    print(f"Embedding {len(texts)} docs...")
    vecs = embed_batch(client, list(texts))
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    for q in QUESTIONS:
        print(f"\nQ: {q}")
        q_vec = embed_batch(client, [q])
        faiss.normalize_L2(q_vec)
        _, ids = index.search(q_vec, k=3)
        retrieved = "\n\n---\n\n".join(texts[i] for i in ids[0])

        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system",
                 "content": "Answer the user's question concisely using ONLY the provided context. "
                            "If the context doesn't cover the question, say so."},
                {"role": "user", "content": f"Context:\n\n{retrieved}\n\nQuestion: {q}"},
            ],
            max_tokens=200,
        )
        print(f"A: {(r.choices[0].message.content or '').strip()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.4: Create the file.**

### Step 8.5: Write `README.md`

```markdown
# 02 - RAG: embeddings + chat from one daemon

**What this shows:** Two model families (a tiny embedding model and a
chat model) coexist on the same daemon. The client talks to two
OpenAI endpoints (`/v1/embeddings` and `/v1/chat/completions`) without
knowing they're served from the same process.

## Setup

```bash
./setup.sh
pip install openai faiss-cpu numpy
```

## Run

```bash
python client.py
```

## What to expect

The script embeds 8 short docs about serve-engine into an in-memory
FAISS index, then asks 4 questions. Each question is embedded, top-3
docs retrieved, and the chat model answers from the retrieved context.

See `sample-output.txt` for a real recorded run.

## Why this matters

Production agent stacks rarely use a single model. Embeddings are
cheap and small; chat is the heavy lifter. serve-engine treats them
uniformly - same daemon, same auth, same metrics endpoint.
```

- [ ] **Step 8.5: Create the file.**

### Step 8.6: Run the recipe end-to-end

```bash
cd examples/02-rag-embed-chat
./setup.sh
python client.py | tee sample-output.txt
cd ../..
```

- [ ] **Step 8.6: Recipe runs cleanly; sample-output.txt captured.**

### Step 8.7: Commit

```bash
git add examples/02-rag-embed-chat/
git commit -m "feat(examples): recipe 02 - RAG with embeddings + chat

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8.7: Commit.**

---

## Task 9: Recipe 03 - LoRA per task

**Pre-task decision:** Identify the base + LoRAs. Open question from the spec.

**Recommended approach:**
- Base: `Qwen/Qwen2.5-1.5B-Instruct` (smaller is fine - the point is the hot-load, not the base capability).
- LoRAs: search HuggingFace for `qwen2.5 1.5b lora` - candidates exist (functional/structured-output adapters, persona adapters, math adapters). Pick 2 with **explicit qwen2.5-1.5B-instruct** in the model card and same target_modules.
- **Fallback:** train two tiny LoRAs ourselves with `peft` (~30 min on a 4090). Host in `Mapika/serve-engine-demo-adapters` or similar.

**Files:**
- Create: `examples/03-lora-per-task/README.md`
- Create: `examples/03-lora-per-task/setup.sh`
- Create: `examples/03-lora-per-task/client.py`
- Create: `examples/03-lora-per-task/sample-output.txt` (after run)

### Step 9.1: Lock the LoRA choice

Find and verify 2 public LoRAs:

```bash
# Replace <REPO_A> and <REPO_B> with the chosen LoRAs.
huggingface-cli download --repo-type adapter <REPO_A> --quiet
huggingface-cli download --repo-type adapter <REPO_B> --quiet
```

Or train two ourselves (placeholder - implementation chooses one path):
- Train a "JSON-output" LoRA: SFT on a tiny synthetic dataset of `{question -> JSON answer}` pairs.
- Train a "concise" LoRA: SFT on a dataset of one-sentence answers.

Either way, end this step with two public LoRA repos that load against Qwen2.5-1.5B-Instruct.

- [ ] **Step 9.1: Two LoRA repos identified and verified to load against the base.**

### Step 9.2: Write `setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Replace <REPO_A> and <REPO_B> with the LoRAs chosen in Step 9.1.
REPO_A="${REPO_A:-Mapika/serve-engine-demo-json}"
REPO_B="${REPO_B:-Mapika/serve-engine-demo-concise}"

serve pull Qwen/Qwen2.5-1.5B-Instruct --name qwen-1_5b
serve run qwen-1_5b --gpu 0 --pin --max-loras 4

serve adapter pull "$REPO_A" --base qwen-1_5b --name json-mode
serve adapter pull "$REPO_B" --base qwen-1_5b --name concise

serve adapter ls
echo
echo "Base + 2 adapters registered. Run: python client.py"
```

- [ ] **Step 9.2: Create + chmod +x. Replace REPO_A/REPO_B with locked names from 9.1.**

### Step 9.3: Write `client.py`

```python
"""Recipe 03 - LoRA per task.

Demonstrates LoRA hot-load: same base model, different adapter per
request, sub-second swap. Client just sets `model=<adapter-name>`.
serve-engine figures out which deployment hosts that adapter and
loads it on demand if it isn't already resident.
"""
from __future__ import annotations

import os
import time

from openai import OpenAI

BASE_URL = os.environ.get("SERVE_URL", "http://127.0.0.1:11500/v1")

TESTS = [
    ("json-mode", "Give me a JSON object representing a user named Ada Lovelace, born 1815."),
    ("concise",   "Why is the sky blue? One sentence only."),
    ("qwen-1_5b", "Why is the sky blue? One sentence only."),  # base, no adapter
    ("json-mode", "Give me a JSON object representing a planet named Mars with three moons."),
    ("concise",   "What is recursion? One sentence only."),
]


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "no-auth"))

    for model, prompt in TESTS:
        t0 = time.monotonic()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        dt = time.monotonic() - t0
        text = (r.choices[0].message.content or "").strip()
        print(f"[{model:>10}] [{dt:>5.2f}s] {text[:100]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.3: Create the file.**

### Step 9.4: Write `README.md`

```markdown
# 03 - LoRA per task

**What this shows:** One base model, multiple task LoRAs, switched
per request via the OpenAI `model` field. The first request to a new
adapter pays a small (~sub-second) hot-load cost; subsequent requests
to that adapter return at the engine's normal latency.

This is the feature most "inference wrappers" don't have.

## Setup

```bash
./setup.sh                    # pulls the base + 2 LoRAs, pins the base
pip install openai
```

## Run

```bash
python client.py
```

## What to expect

5 requests:
- The first two target two different adapters - each shows a one-time hot-load cost in the timing column.
- The third hits the bare base model.
- The last two re-target the previously-loaded adapters - these are fast.

See `sample-output.txt`.

## Why this matters

Per-tenant LoRAs are a common ask: "each customer / each agent / each
team has their fine-tuned adapter, but we want to serve them from one
GPU without restarting." serve-engine's `adapter` subsystem handles
that with sub-second swaps. No restart, no second deployment, one
OpenAI endpoint.
```

- [ ] **Step 9.4: Create the file.**

### Step 9.5: Run end-to-end

```bash
cd examples/03-lora-per-task
./setup.sh
python client.py | tee sample-output.txt
cd ../..
```

- [ ] **Step 9.5: Recipe runs; sample-output.txt captured. Verify timing column shows the hot-load pattern (first request to each new adapter slow, subsequent fast).**

### Step 9.6: Commit

```bash
git add examples/03-lora-per-task/
git commit -m "feat(examples): recipe 03 - LoRA per task (hot-load)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9.6: Commit.**

---

## Task 10: README rewrite

**Files:**
- Modify: `README.md`

### Step 10.1: Rewrite the first paragraph

Replace lines 1-7 of `README.md` (the title, tagline, and Status line) with:

```markdown
# serve-engine

The GPU box for a small AI team. Your agents call 5 models from one
OpenAI endpoint - a tiny router, a medium retriever, a big reasoner,
plus the LoRAs your team trained yesterday - on one machine.
serve-engine keeps the hot ones warm and swaps the rest.

A single-node multi-user inference orchestrator over vLLM, SGLang,
and TensorRT-LLM. OpenAI-compatible HTTP, real rate limits, sub-second
LoRA hot-load, persistent torch.compile cache for fast warm restarts.

**Status:** core lifecycle, observability, auth, three engine backends
(vLLM, SGLang, TensorRT-LLM), and the v2 stack (adapters / snapshots /
predictor) all wired and verified on H100 and RTX PRO 6000 Blackwell.
363 unit + integration tests, ruff clean.
```

- [ ] **Step 10.1: Replace the opening section.**

### Step 10.2: Reorder "What it does" - agent-relevant bullets first

In the existing "What it does" section, reorder the bullets to this sequence:

1. **One daemon, many models.** (existing)
2. **LoRA hot-load.** (new bullet - describe sub-second adapter swap; reference recipe 03)
3. **OpenAI-compatible.** (existing)
4. **Real rate limits.** (existing)
5. **Engine pluggability.** (moved down - important but technical)
6. **Crash-safe.** (existing)
7. **Observable.** (existing)
8. **Web UI.** (existing)
9. **Bootstrap-friendly.** (existing)

Write the new LoRA bullet to insert after "One daemon, many models":

```markdown
- **LoRA hot-load.** Pull adapters from HuggingFace with `serve adapter pull`,
  then OpenAI requests with `model=<adapter-name>` are transparently routed
  to the right base with sub-second hot-swap. See `examples/03-lora-per-task/`.
```

- [ ] **Step 10.2: Reorder the bullets + insert the LoRA line.**

### Step 10.3: Add a "Cold start" section under "Tested performance"

After the existing "Tested performance" table (around line 169 of the current README), insert:

```markdown
### Cold start vs. warm restart

Single H100 80GB, `Qwen/Qwen2.5-1.5B-Instruct` on vLLM. 5 cold + 5 warm
runs; medians shown. Raw numbers in `docs/bench/snapshot-cold-vs-warm.json`,
re-runnable with `scripts/bench_snapshots.py`.

<!-- PASTE the markdown table emitted by Task 5's bench run here. -->

The cold run pays the torch.compile compile cost; subsequent runs reuse
the cached kernels from `~/.serve/snapshots/<key>/`. **First boot of
any new model is always cold** - the snapshot is built during that
first run. Every subsequent boot of the same configuration is warm.

The same mechanism applies to SGLang; benchmarking is identical.
```

Then paste the **actual markdown table** from `docs/bench/snapshot-cold-vs-warm.json`'s stdout output (captured in Task 5.2).

- [ ] **Step 10.3: Insert the section + paste the real bench table.**

### Step 10.4: Add an "Examples" section

After the "Tested performance" section, insert:

```markdown
## Examples

Three self-contained recipes under [`examples/`](examples/):

- [`01-router-reasoner/`](examples/01-router-reasoner/) - Cheap routing with a small model + a medium reasoner. Two models, one endpoint, real cost savings.
- [`02-rag-embed-chat/`](examples/02-rag-embed-chat/) - RAG over serve-engine's own README. Embeddings + chat from one daemon.
- [`03-lora-per-task/`](examples/03-lora-per-task/) - LoRA hot-load. Same base model, different adapter per request, sub-second swap.
```

- [ ] **Step 10.4: Insert the Examples section.**

### Step 10.5: Verify the README renders + commit

```bash
# Eyeball the README - open in editor or render preview
git diff README.md
```

- [ ] **Step 10.5: Sanity-check the rewrite. No broken tables, no orphaned `<!--PASTE-->` markers.**

```bash
git add README.md
git commit -m "docs(readme): agent-stack positioning + cold-start benchmark

- Opening paragraph leads with the agent narrative.
- LoRA hot-load promoted into the top capability list.
- New 'Cold start vs. warm restart' subsection with the real
  bench numbers from docs/bench/snapshot-cold-vs-warm.json.
- New 'Examples' section linking to the three recipes.

Existing architecture / on-disk / CLI / quickstart sections unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 10.5: Commit.**

---

## Task 11: `LAUNCH.md` draft

**Files:**
- Create: `LAUNCH.md`

### Step 11.1: Write the launch doc

```markdown
# Launch - serve-engine open-source

**Status:** Draft. Edit this in place - it's the source of truth for
the launch copy, HN post, and any external write-ups.

## The 200-word version (HN top-of-comment / Twitter thread / Reddit)

serve-engine is a single-node inference orchestrator for small AI teams
running agent workloads. One daemon serves 5 models from one
OpenAI-compatible endpoint - a tiny router, a medium retriever, a big
reasoner, plus task LoRAs your team trained yesterday - on one machine.
Hot models stay resident, cold ones swap on demand.

Three things worth a closer look:

1. **Sub-second LoRA hot-load.** Pull adapters from HuggingFace, hit
   the API with `model=<adapter>`, get sub-second swaps. See the
   `examples/03-lora-per-task/` recipe.

2. **Snapshot-warm restarts.** First boot of a new model pays the
   torch.compile cost. Every subsequent boot reuses the cached
   kernels - `<INSERT_TTFT_NUMBER>` on Qwen2.5-1.5B vs.
   `<INSERT_COLD_NUMBER>` cold. Numbers and a re-runnable benchmark
   in the repo.

3. **vLLM + SGLang + TensorRT-LLM under one API.** Same
   `/v1/chat/completions` regardless of which engine is behind it.

Apache-2.0. 363 tests. Built for the team that has a GPU box, not a
GPU fleet. <REPO_URL>

## The 600-word blog post

[Write the long-form version here once the 200-word version is locked.
Aim for: the problem (small AI team has a GPU and wants to serve
multiple models without standing up Kubernetes), the existing options
(vllm serve, LiteLLM, BentoML - what each is missing for this
audience), the serve-engine answer with the three differentiators
above, then honest limits (single-node only, no multi-node, no
autotune, no built-in TLS).]

## Launch-day checklist

- [ ] Push branch to `main` on github.com/<USER>/serve-engine
- [ ] Tag a v0.1.0 release (update `pyproject.toml` version, `__version__`)
- [ ] Post HN with title: "Show HN: serve-engine - multi-model inference orchestrator for one GPU box"
- [ ] Top-of-comment: paste the 200-word version above
- [ ] Reddit /r/LocalLLaMA: longer post with the cold-start table screenshot
- [ ] Twitter/X thread: 5 tweets - problem, hot-load demo, cold-start numbers, repo link, ask for feedback
- [ ] Monitor for 4 hours; respond to comments

## Out of scope for launch day

- Multi-node distributed serving
- Autotune (model -> optimal TP / dtype)
- Built-in TLS
- Contributor-onboarding docs (the project is single-author for now;
  PRs welcome but no formal CONTRIBUTING.md yet)
```

- [ ] **Step 11.1: Create `LAUNCH.md`.** Before committing, substitute the actual `<INSERT_TTFT_NUMBER>` / `<INSERT_COLD_NUMBER>` placeholders with the median values from `docs/bench/snapshot-cold-vs-warm.json` (Task 5 output).

### Step 11.2: Commit

```bash
git add LAUNCH.md
git commit -m "docs: launch-day draft (HN copy + blog skeleton + checklist)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 11.2: Commit.**

---

## Final verification

### Step F.1: Full test suite + lint

```bash
pytest -q
ruff check src/ tests/ scripts/
```

Expected: all tests pass (364+ - original 363 + the new bench unit tests), ruff clean.

- [ ] **Step F.1: Green.**

### Step F.2: All three recipes still run

For each recipe, re-run `./setup.sh && python client.py` and confirm the output still matches `sample-output.txt` shape (not exact strings - token sampling varies - but program completes, expected sections present).

- [ ] **Step F.2: All three recipes runnable from scratch.**

### Step F.3: Final commit log review

```bash
git log --oneline main..HEAD
```

Should show ~11-12 commits, one per task, each green.

- [ ] **Step F.3: Commit log clean.**
