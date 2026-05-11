from __future__ import annotations

from typing import ClassVar

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.lifecycle.plan import DeploymentPlan

INTERNAL_PORT = 30000


class SGLangBackend:
    name: ClassVar[str] = "sglang"
    image_default: ClassVar[str] = "lmsysorg/sglang:v0.5.5.post1"
    health_path: ClassVar[str] = "/health"
    openai_base: ClassVar[str] = "/v1"
    metrics_path: ClassVar[str] = "/metrics"
    internal_port: ClassVar[int] = INTERNAL_PORT

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        argv: list[str] = [
            "--model-path", local_model_path,
            "--tp", str(plan.tensor_parallel),
            "--context-length", str(plan.max_model_len),
            "--mem-fraction-static", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype if plan.dtype != "auto" else "auto",
            "--host", "0.0.0.0",
            "--port", str(INTERNAL_PORT),
            "--served-model-name", plan.model_name,
        ]
        for k, v in plan.extra_args.items():
            argv.extend([k, v])
        return argv

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
