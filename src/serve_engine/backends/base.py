from __future__ import annotations

from typing import ClassVar, Protocol

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.backends.manifest import EngineManifest, Headroom, load_manifest
from serve_engine.lifecycle.plan import DeploymentPlan


class Backend(Protocol):
    name: str
    image_default: str
    health_path: str
    openai_base: str
    metrics_path: str
    internal_port: int
    headroom: Headroom

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]: ...
    def container_env(self, plan: DeploymentPlan) -> dict[str, str]: ...
    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]: ...


class ContainerBackend:
    """Shared scaffolding for engines that run as Docker containers with
    NVIDIA device requests. Subclasses set `name` and implement `build_argv`."""

    name: ClassVar[str]

    def __init__(self, manifest: EngineManifest | None = None):
        if manifest is None:
            manifest = load_manifest()[self.name]
        self.manifest = manifest

    @property
    def image_default(self) -> str:
        return self.manifest.image_default

    @property
    def health_path(self) -> str:
        return self.manifest.health_path

    @property
    def openai_base(self) -> str:
        return self.manifest.openai_base

    @property
    def metrics_path(self) -> str:
        return self.manifest.metrics_path

    @property
    def internal_port(self) -> int:
        return self.manifest.internal_port

    @property
    def headroom(self) -> Headroom:
        return self.manifest.headroom

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        raise NotImplementedError

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

    @staticmethod
    def _append_extra(argv: list[str], extra: dict[str, str]) -> None:
        """Append user-provided extra_args from the deployment plan.
        Empty value means bare flag (e.g. --enable-expert-parallel)."""
        for k, v in extra.items():
            if v == "":
                argv.append(k)
            else:
                argv.extend([k, v])
