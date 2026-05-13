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
