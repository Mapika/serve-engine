from __future__ import annotations

from serve_engine.backends.base import ContainerBackend
from serve_engine.lifecycle.plan import DeploymentPlan


class SGLangBackend(ContainerBackend):
    name = "sglang"

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
            "--dtype", plan.dtype,
            "--host", "0.0.0.0",
            "--port", str(self.manifest.internal_port),
            "--served-model-name", plan.model_name,
        ]
        argv.extend(self.manifest.extra_launch_args)
        self._append_extra(argv, plan.extra_args)
        return argv
