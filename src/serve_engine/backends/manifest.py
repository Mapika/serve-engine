from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Headroom:
    factor: float
    min_extra_mb: int
    min_floor_pct: int

    def effective_util(self, *, reserved_mb: int, per_gpu_mb: int) -> float:
        """Compute the gpu_memory_utilization fraction to hand to the engine."""
        target = max(
            reserved_mb * self.factor,
            reserved_mb + self.min_extra_mb,
            per_gpu_mb * (self.min_floor_pct / 100.0),
        )
        return min(0.95, max(0.05, target / per_gpu_mb))


@dataclass(frozen=True)
class EngineManifest:
    name: str
    image: str
    pinned_tag: str
    health_path: str
    openai_base: str
    metrics_path: str
    internal_port: int
    headroom: Headroom

    @property
    def image_default(self) -> str:
        return f"{self.image}:{self.pinned_tag}"


def load_manifest(path: Path | None = None) -> dict[str, EngineManifest]:
    if path is None:
        text = files("serve_engine.backends").joinpath("backends.yaml").read_text()
    else:
        text = Path(path).read_text()
    raw = yaml.safe_load(text) or {}
    out: dict[str, EngineManifest] = {}
    for name, e in raw.items():
        hr = e.get("headroom") or {}
        out[name] = EngineManifest(
            name=name,
            image=e["image"],
            pinned_tag=e["pinned_tag"],
            health_path=e.get("health_path", "/health"),
            openai_base=e.get("openai_base", "/v1"),
            metrics_path=e.get("metrics_path", "/metrics"),
            internal_port=int(e["internal_port"]),
            headroom=Headroom(
                factor=float(hr.get("factor", 1.5)),
                min_extra_mb=int(hr.get("min_extra_mb", 2048)),
                min_floor_pct=int(hr.get("min_floor_pct", 15)),
            ),
        )
    return out
