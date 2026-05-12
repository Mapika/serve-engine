from __future__ import annotations

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.backends.manifest import EngineManifest
from serve_engine.lifecycle.plan import DeploymentPlan


class VLLMBackend:
    name = "vllm"

    def __init__(self, manifest: EngineManifest | None = None):
        if manifest is None:
            from serve_engine.backends.manifest import load_manifest
            manifest = load_manifest()["vllm"]
        self._m = manifest

    @property
    def image_default(self) -> str:
        return self._m.image_default

    @property
    def health_path(self) -> str:
        return self._m.health_path

    @property
    def openai_base(self) -> str:
        return self._m.openai_base

    @property
    def metrics_path(self) -> str:
        return self._m.metrics_path

    @property
    def internal_port(self) -> int:
        return self._m.internal_port

    @property
    def headroom(self):
        return self._m.headroom

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        argv: list[str] = [
            "--model", local_model_path,
            "--tensor-parallel-size", str(plan.tensor_parallel),
            "--max-model-len", str(plan.max_model_len),
            "--max-num-seqs", str(plan.target_concurrency),
            "--gpu-memory-utilization", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype,
            "--host", "0.0.0.0",
            "--port", str(self._m.internal_port),
            "--served-model-name", plan.model_name,
        ]
        if plan.enable_prefix_caching:
            argv.append("--enable-prefix-caching")
        if plan.enable_chunked_prefill:
            argv.append("--enable-chunked-prefill")
        for k, v in plan.extra_args.items():
            # Empty value = bare flag (e.g. --enable-expert-parallel).
            if v == "":
                argv.append(k)
            else:
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
