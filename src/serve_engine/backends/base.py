from __future__ import annotations

from typing import Protocol

from serve_engine.lifecycle.plan import DeploymentPlan


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
