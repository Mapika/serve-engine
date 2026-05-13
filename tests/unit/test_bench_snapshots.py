from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from scripts.bench_snapshots import (
    RunResult,
    aggregate,
    gpu_fingerprint,
    to_markdown,
)


@asynccontextmanager
async def fake_daemon(tmp_path):
    """In-process FastAPI server on a UDS socket. Simulates the admin
    surface bench_snapshots needs: model registration, deployment
    create / list / delete, plus a fake streaming chat completion."""
    app = FastAPI()
    state = {"next_id": 1, "deps": {}, "poll_count": {}}

    @app.post("/admin/models", status_code=201)
    async def create_model(body: dict):
        return {"name": body["name"]}

    @app.post("/admin/deployments", status_code=201)
    async def create_dep(body: dict):
        did = state["next_id"]
        state["next_id"] += 1
        state["deps"][did] = {
            "id": did, "status": "loading", "model_name": body["model_name"],
        }
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
        async def gen():
            await asyncio.sleep(0.01)
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")

    sock = tmp_path / "fake.sock"
    config = uvicorn.Config(app, uds=str(sock), log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(50):
        if sock.exists():
            break
        await asyncio.sleep(0.02)
    try:
        yield sock
    finally:
        server.should_exit = True
        with suppress(asyncio.CancelledError, asyncio.TimeoutError):
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
    assert result.ok is True, f"run failed: {result.error}"
    assert result.error is None
    assert result.ready_s > 0
    assert result.ttft_s > 0
    assert result.total_s >= result.ready_s + result.ttft_s - 0.001  # rounding


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
    assert "5.8" in md or "5.83" in md  # 47.2 / 8.1 ≈ 5.83
    assert "36" in md  # 12.4 / 0.34 ≈ 36.5
