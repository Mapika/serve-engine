from __future__ import annotations

from dataclasses import dataclass

try:
    import pynvml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GPUSnapshot:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    gpu_util_pct: int
    power_w: int


def read_gpu_stats() -> list[GPUSnapshot]:
    """Live per-GPU memory + utilization + power. Empty list if pynvml absent."""
    if pynvml is None:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    out: list[GPUSnapshot] = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
        except Exception:
            power_mw = 0
        out.append(GPUSnapshot(
            index=i,
            memory_used_mb=int(mem.used) // 1024 // 1024,
            memory_total_mb=int(mem.total) // 1024 // 1024,
            gpu_util_pct=int(util.gpu),
            power_w=int(power_mw) // 1000,
        ))
    return out
