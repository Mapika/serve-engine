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

    def build_argv(
        self, plan: DeploymentPlan, *, local_model_path: str, config_path: str | None = None,
    ) -> list[str]: ...
    def container_env(self, plan: DeploymentPlan) -> dict[str, str]: ...
    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]: ...
    def engine_config(self, plan: DeploymentPlan) -> dict | None: ...


class ContainerBackend:
    """Shared scaffolding for engines that run as Docker containers with
    NVIDIA device requests. Subclasses set `name` and implement `build_argv`."""

    name: ClassVar[str]
    # True if this backend can hot-load LoRA adapters at runtime via the
    # engine's load/unload HTTP endpoints. vLLM and SGLang inherit True;
    # TRT-LLM overrides to False (its adapter story is on the legacy
    # AOT-engine path, incompatible with our PyTorch-backend deployments).
    supports_adapters: ClassVar[bool] = True
    # Per-engine HTTP paths for dynamic adapter load/unload. vLLM uses
    # /v1/* paths; SGLang uses bare paths. Override per backend.
    adapter_load_path: ClassVar[str] = "/v1/load_lora_adapter"
    adapter_unload_path: ClassVar[str] = "/v1/unload_lora_adapter"
    # Sub-project B (snapshots). True if a previously-built engine state
    # (torch.compile cache, cuda graphs, …) can be persisted on disk and
    # reused by a subsequent deployment with the same snapshot_key.
    # vLLM and SGLang override to True; TRT-LLM stays False until its
    # PyTorch-backend snapshot story matures.
    supports_snapshots: ClassVar[bool] = False

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

    def build_argv(
        self,
        plan: DeploymentPlan,
        *,
        local_model_path: str,
        config_path: str | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def engine_config(self, plan: DeploymentPlan) -> dict | None:
        """Optional per-deployment YAML config for the engine.

        Backends that support a `--config <file>` flag (or equivalent) can
        return a dict here; the manager serializes it to YAML, mounts it
        into the container, and passes the in-container path to build_argv
        as `config_path=`. Returning None means no config file is needed.
        """
        return None

    def container_env(self, plan: DeploymentPlan) -> dict[str, str]:
        return {}

    # ---- Snapshot hooks (Sub-project B) ----
    #
    # vLLM and SGLang both build their warm-up state (torch.compile
    # inductor cache) in a directory pointed to by TORCHINDUCTOR_CACHE_DIR.
    # We bind-mount ~/.serve/snapshots/<key>/ into the container at
    # /snapshots and point the env var at /snapshots/torch_cache —
    # inductor populates it on first run, finds the cached kernels on
    # subsequent runs of the same shape. "Saving" reduces to sampling
    # the host-side directory size afterward (handled by the manager).
    #
    # The default implementations below describe the inductor-cache
    # approach; backends opt in by flipping supports_snapshots to True
    # (vLLM, SGLang). TRT-LLM keeps the default False and the hooks
    # therefore return empty — no snapshot work happens.
    SNAPSHOT_MOUNT_PATH: ClassVar[str] = "/snapshots"

    def snapshot_mount(self, snapshot_path: str) -> dict[str, dict]:
        """Bind-mounts to add when a snapshot is in play. host_path →
        {"bind": container_path, "mode": "rw"} — matches docker-py's
        container_kwargs `volumes` shape."""
        if not self.supports_snapshots:
            return {}
        return {snapshot_path: {"bind": self.SNAPSHOT_MOUNT_PATH, "mode": "rw"}}

    def snapshot_load_argv(self, snapshot_path: str) -> list[str]:
        """Engine flags to add when a snapshot is in play. Default: none;
        the env-var path covers both vLLM and SGLang. Reserved for
        engines that need an explicit --load-from flag in the future."""
        return []

    def snapshot_env(self, snapshot_path: str) -> dict[str, str]:
        """Env vars to set when a snapshot is in play. Default: empty.

        The compile-cache env var is engine-specific — vLLM ignores
        TORCHINDUCTOR_CACHE_DIR for its own compile cache and uses
        VLLM_CACHE_ROOT, SGLang uses TORCHINDUCTOR_CACHE_DIR. Backends
        override this to point at the bind-mount.
        """
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

        Empty value means bare flag (e.g. --enable-expert-parallel).

        If the backend already emitted the same flag, the earlier emission
        (and its value) is removed first so the user override wins cleanly
        instead of relying on argparse last-value-wins semantics — strict
        argparsers (some TRT-LLM versions) reject duplicate flags outright.
        Heuristic for "next token is the flag's value": doesn't start with
        '--'. Engine flag values are paths/numbers/dtype names; collisions
        are not a real concern.
        """
        for k, v in extra.items():
            i = 0
            while i < len(argv):
                if argv[i] == k:
                    if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                        del argv[i:i + 2]
                    else:
                        del argv[i]
                else:
                    i += 1
            if v == "":
                argv.append(k)
            else:
                argv.extend([k, v])
