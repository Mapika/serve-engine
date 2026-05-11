from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SUPPORTED_BACKENDS = ("vllm",)  # Plan 04 adds "sglang"
SUPPORTED_DTYPES = ("auto", "bf16", "fp16", "fp8")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class DeploymentPlan:
    model_name: str
    hf_repo: str
    revision: str
    backend: Literal["vllm"]
    image_tag: str
    gpu_ids: list[int]
    max_model_len: int
    tensor_parallel: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    extra_args: dict[str, str] = field(default_factory=dict)
    pinned: bool = False
    idle_timeout_s: int | None = None
    target_concurrency: int = 8  # used by KV estimator

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend {self.backend!r} not supported in Plan 01 "
                f"(supported: {SUPPORTED_BACKENDS})"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype {self.dtype!r} not in {SUPPORTED_DTYPES}")
        if self.tensor_parallel < 1:
            raise ValueError("tensor_parallel must be >= 1")
        if not _is_power_of_two(self.tensor_parallel):
            raise ValueError("tensor_parallel must be a power of 2")
        if self.tensor_parallel != len(self.gpu_ids):
            raise ValueError(
                "tensor_parallel must equal len(gpu_ids); "
                f"got TP={self.tensor_parallel}, gpus={self.gpu_ids}"
            )
        if not 0.1 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in [0.1, 1.0]")
