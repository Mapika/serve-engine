from __future__ import annotations

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.backends.manifest import EngineManifest
from serve_engine.lifecycle.plan import DeploymentPlan


class SGLangBackend:
    name = "sglang"

    def __init__(self, manifest: EngineManifest | None = None):
        if manifest is None:
            from serve_engine.backends.manifest import load_manifest
            manifest = load_manifest()["sglang"]
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
        # The lmsysorg/sglang image's ENTRYPOINT is nvidia_entrypoint.sh which
        # `exec`s its argv. Unlike vLLM's image, it does NOT embed the launcher,
        # so we must include `python3 -m sglang.launch_server` ourselves.
        argv: list[str] = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", local_model_path,
            "--tp", str(plan.tensor_parallel),
            "--context-length", str(plan.max_model_len),
            "--max-running-requests", str(plan.target_concurrency),
            "--mem-fraction-static", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype if plan.dtype != "auto" else "auto",
            "--host", "0.0.0.0",
            "--port", str(self._m.internal_port),
            "--served-model-name", plan.model_name,
            # Piecewise CUDA graph compilation in v0.5.11 hit CUBLAS_STATUS_EXECUTION_FAILED
            # on small models. Disabling falls back to regular CUDA graphs (still fast).
            # Re-evaluate once a fixed SGLang release ships.
            "--disable-piecewise-cuda-graph",
        ]
        for k, v in plan.extra_args.items():
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
