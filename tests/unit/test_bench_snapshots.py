from __future__ import annotations

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
    assert "5.8" in md or "5.83" in md  # 47.2 / 8.1 ≈ 5.83
    assert "36" in md  # 12.4 / 0.34 ≈ 36.5
