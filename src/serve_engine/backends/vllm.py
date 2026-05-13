from __future__ import annotations

from serve_engine.backends.base import ContainerBackend
from serve_engine.lifecycle.plan import DeploymentPlan


class VLLMBackend(ContainerBackend):
    name = "vllm"
    # Persist the engine's torch.compile cache across deployments. vLLM
    # 0.20.2 ignores TORCHINDUCTOR_CACHE_DIR for its own compile cache —
    # it writes under VLLM_CACHE_ROOT/torch_compile_cache/. Pointing
    # VLLM_CACHE_ROOT at our bind-mounted /snapshots directory means
    # the next deployment with the same shape finds the compiled
    # kernels already on disk and skips graph compilation (~30-60s
    # → ~5-15s on first-token).
    supports_snapshots = True

    def snapshot_env(self, snapshot_path: str) -> dict[str, str]:
        if not self.supports_snapshots:
            return {}
        return {"VLLM_CACHE_ROOT": self.SNAPSHOT_MOUNT_PATH}

    def container_env(self, plan: DeploymentPlan) -> dict[str, str]:
        env = super().container_env(plan)
        if plan.max_loras > 0:
            # vLLM's /v1/load_lora_adapter is gated behind this env var by
            # default for safety. We unlock it whenever the deployment
            # opted into LoRA — the daemon owns access to the endpoint
            # behind admin-only auth, so external misuse isn't a concern.
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
        return env

    def build_argv(
        self,
        plan: DeploymentPlan,
        *,
        local_model_path: str,
        config_path: str | None = None,  # vLLM doesn't use a YAML config file
    ) -> list[str]:
        argv: list[str] = [
            "--model", local_model_path,
            "--tensor-parallel-size", str(plan.tensor_parallel),
            "--max-model-len", str(plan.max_model_len),
            "--max-num-seqs", str(plan.target_concurrency),
            "--gpu-memory-utilization", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype,
            "--host", "0.0.0.0",
            "--port", str(self.manifest.internal_port),
            "--served-model-name", plan.model_name,
        ]
        if plan.enable_prefix_caching:
            argv.append("--enable-prefix-caching")
        if plan.enable_chunked_prefill:
            argv.append("--enable-chunked-prefill")
        if plan.max_loras > 0:
            # LoRA hot-load via /v1/load_lora_adapter endpoint requires
            # --enable-lora at startup. --max-loras caps the concurrent
            # adapter slots; adapters beyond this evict each other (per
            # the per-deployment LRU we maintain in deployment_adapters).
            argv.extend([
                "--enable-lora",
                "--max-loras", str(plan.max_loras),
            ])
        argv.extend(self.manifest.extra_launch_args)
        self._append_extra(argv, plan.extra_args)
        return argv
