from __future__ import annotations

from serve_engine.backends.base import ContainerBackend
from serve_engine.lifecycle.plan import DeploymentPlan


class TRTLLMBackend(ContainerBackend):
    """NVIDIA TensorRT-LLM via `trtllm-serve` (PyTorch backend).

    Flag naming convention is snake_case (--tp_size, not --tensor-parallel-size).
    The TensorRT-engine path requires an ahead-of-time `trtllm-build` step and
    is not exposed here — use the PyTorch backend for parity with vLLM/SGLang.
    """

    name = "trtllm"

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        # The nvcr.io/nvidia/tensorrt-llm/release image does not embed the
        # launcher as ENTRYPOINT, so we invoke `trtllm-serve` explicitly.
        # Model path is positional, immediately after the launcher.
        argv: list[str] = [
            "trtllm-serve", local_model_path,
            "--backend", "pytorch",
            "--host", "0.0.0.0",
            "--port", str(self.manifest.internal_port),
            "--tp_size", str(plan.tensor_parallel),
            "--max_seq_len", str(plan.max_model_len),
            "--max_batch_size", str(plan.target_concurrency),
            "--kv_cache_free_gpu_memory_fraction", str(plan.gpu_memory_utilization),
            "--trust_remote_code",
        ]
        # TRT-LLM does not accept --served_model_name on the CLI; OpenAI
        # `model` field is matched against the model path/repo. Clients should
        # pass the same string they registered with.
        argv.extend(self.manifest.extra_launch_args)
        self._append_extra(argv, plan.extra_args)
        return argv
