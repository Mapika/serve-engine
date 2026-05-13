#!/usr/bin/env python3
"""Cold-start snapshot benchmark for serve-engine.

Measures the wallclock cost of bringing a model from "not loaded" to
"first inference token" — once with no snapshot (cold), once with a
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


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
    mem_digits = "".join(c for c in parts[2] if c.isdigit())
    return {
        "name": parts[0],
        "driver": parts[1],
        "memory_mib": int(mem_digits) if mem_digits else 0,
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
        return f"{c / w:.1f}×" if w > 0 else "n/a"

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


class AdminClient:
    """Thin async client over the daemon UDS socket. UDS path bypasses
    admin auth (the daemon trusts anything on the socket — same trust
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
        4xx — that's fine, we swallow it."""
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

    async def wait_ready(
        self, dep_id: int, poll_interval_s: float, timeout_s: float = 600.0,
    ) -> None:
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
                    raise RuntimeError(
                        f"deployment {dep_id} failed: {d.get('last_error')}",
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"deployment {dep_id} not ready after {timeout_s}s",
                    )
                await asyncio.sleep(poll_interval_s)

    async def first_ttft(self, model_name: str) -> float:
        """Send a chat completion, measure wallclock to first byte of the
        first SSE token chunk. Streams so we genuinely measure TTFT,
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
    """One full timed cycle: create deployment → wait ready → measure TTFT → delete."""
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
