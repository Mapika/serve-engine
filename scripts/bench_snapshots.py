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
import shutil
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


def _purge_snapshots(snapshots_dir: Path) -> None:
    """Delete all snapshot directories. Used before each cold run.
    We delete the whole dir contents rather than guessing snapshot_keys
    because the key is content-addressable on the deployment config —
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
    snapshot_save_timeout_s: float = 120.0,
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
            await _wait_for_snapshot_save(snapshots_dir, snapshot_save_timeout_s)

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF repo id; also used as the registered model name (with / → -).",
    )
    p.add_argument("--runs", type=int, default=5)
    p.add_argument(
        "--output", type=Path,
        default=Path("docs/bench/snapshot-cold-vs-warm.json"),
    )
    p.add_argument("--sock", type=Path, default=Path.home() / ".serve" / "sock")
    p.add_argument(
        "--snapshots-dir", type=Path,
        default=Path.home() / ".serve" / "snapshots",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args()

    if not args.sock.exists():
        print(
            f"error: daemon socket not found at {args.sock} — is the daemon running?",
            file=sys.stderr,
        )
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
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    payload = {
        "model": hf_repo,
        "engine": "vllm",
        "runs": args.runs,
        "hardware": gpu_fingerprint(),
        "serve_engine_git_sha": git_sha,
        "cold": {
            "runs": [asdict(r) for r in result["cold"]],
            "aggregate": cold_agg,
        },
        "warm": {
            "runs": [asdict(r) for r in result["warm"]],
            "aggregate": warm_agg,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")

    # Non-zero exit if too many runs failed — caller can gate launch on this.
    if (
        cold_agg["n_failed"] > args.runs // 2
        or warm_agg["n_failed"] > args.runs // 2
    ):
        print("FAILED: majority of runs errored", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
