from __future__ import annotations

from typing import ClassVar

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.lifecycle.plan import DeploymentPlan

ENGINE_INTERNAL_PORT = 8000


class VLLMBackend:
    name: ClassVar[str] = "vllm"
    image_default: ClassVar[str] = "vllm/vllm-openai:v0.20.2"
    health_path: ClassVar[str] = "/health"
    openai_base: ClassVar[str] = "/v1"
    metrics_path: ClassVar[str] = "/metrics"
    internal_port: ClassVar[int] = ENGINE_INTERNAL_PORT

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        argv: list[str] = [
            "--model", local_model_path,
            "--tensor-parallel-size", str(plan.tensor_parallel),
            "--max-model-len", str(plan.max_model_len),
            "--gpu-memory-utilization", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype,
            "--host", "0.0.0.0",
            "--port", str(ENGINE_INTERNAL_PORT),
            "--served-model-name", plan.model_name,
        ]
        if plan.enable_prefix_caching:
            argv.append("--enable-prefix-caching")
        if plan.enable_chunked_prefill:
            argv.append("--enable-chunked-prefill")
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
