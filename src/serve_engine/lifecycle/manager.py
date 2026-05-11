from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import replace
from pathlib import Path

import httpx

from serve_engine.backends.base import Backend
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.lifecycle.downloader import download_model
from serve_engine.lifecycle.kv_estimator import KVEstimateInput, estimate_vram_mb
from serve_engine.lifecycle.placement import (
    AllocatedDeployment,
    EvictThenFit,
    NoRoom,
    PlacementRequest,
    plan_placement,
)
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.lifecycle.topology import Topology
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store

log = logging.getLogger(__name__)


async def download_model_async(**kwargs) -> str:
    # snapshot_download is blocking; offload to a thread
    return await asyncio.to_thread(download_model, **kwargs)


async def wait_healthy(url: str, *, timeout_s: float = 600.0, interval_s: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while loop.time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(interval_s)
    return False


class LifecycleManager:
    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        docker_client: DockerClient,
        backends: dict[str, Backend],
        models_dir: Path,
        topology: Topology | None = None,
        load_timeout_s: float = 600.0,
    ):
        self._conn = conn
        self._docker = docker_client
        self._backends = backends
        self._models_dir = models_dir
        self._topology = topology
        self._load_timeout_s = load_timeout_s
        self._lock = asyncio.Lock()

    @property
    def docker(self) -> DockerClient:
        return self._docker

    @property
    def active(self):
        # Backwards-compat property: returns the most-recently-loaded ready dep
        # or None. Used by the logs admin endpoint.
        ready = dep_store.list_ready(self._conn)
        return ready[-1] if ready else None

    async def load(self, plan: DeploymentPlan):
        async with self._lock:
            # 1. Ensure model row
            model = model_store.get_by_name(self._conn, plan.model_name)
            if model is None:
                model = model_store.add(
                    self._conn,
                    name=plan.model_name,
                    hf_repo=plan.hf_repo,
                    revision=plan.revision,
                )

            # 2. Ensure weights are local
            local_path = model.local_path
            if local_path is None:
                local_path = await download_model_async(
                    hf_repo=plan.hf_repo,
                    revision=plan.revision,
                    cache_dir=self._models_dir,
                )
                model_store.set_local_path(self._conn, model.id, local_path)

            # 3. Estimate VRAM
            vram_mb = estimate_vram_mb(KVEstimateInput(
                model_dir=Path(local_path),
                max_model_len=plan.max_model_len,
                target_concurrency=plan.target_concurrency,
                dtype=plan.dtype,
            ))

            # 4. Placement
            if self._topology is None:
                raise RuntimeError(
                    "topology not initialized; "
                    "pass topology=read_topology() to LifecycleManager"
                )
            ready = dep_store.list_ready(self._conn)
            # Map id -> LRU rank (lower rank = more evictable). Pinned are absent.
            lru_rank = {
                d.id: idx
                for idx, d in enumerate(dep_store.list_evictable(self._conn))
            }
            allocated = sorted(
                [
                    AllocatedDeployment(
                        id=d.id,
                        gpu_ids=d.gpu_ids,
                        vram_reserved_mb=d.vram_reserved_mb,
                        pinned=d.pinned,
                    )
                    for d in ready
                ],
                # Pinned last (never evicted); within auto, LRU rank ascending
                key=lambda a: (a.pinned, lru_rank.get(a.id, 0)),
            )
            request = PlacementRequest(
                tensor_parallel=plan.tensor_parallel,
                vram_reserved_mb=vram_mb,
                model_name=plan.model_name,
            )
            decision = plan_placement(self._topology, allocated=allocated, request=request)

            if isinstance(decision, NoRoom):
                raise RuntimeError(decision.reason)
            if isinstance(decision, EvictThenFit):
                for victim_id in decision.evict_ids:
                    await self._stop_locked(victim_id)
                gpu_ids = decision.gpu_ids
            else:  # Fit
                gpu_ids = decision.gpu_ids

            # 5. Create row + spawn container
            dep = dep_store.create(
                self._conn,
                model_id=model.id,
                backend=plan.backend,
                image_tag=plan.image_tag,
                gpu_ids=gpu_ids,
                tensor_parallel=len(gpu_ids),
                max_model_len=plan.max_model_len,
                dtype=plan.dtype,
                pinned=plan.pinned,
                idle_timeout_s=plan.idle_timeout_s,
                vram_reserved_mb=vram_mb,
            )
            dep_store.update_status(self._conn, dep.id, "loading")

            backend = self._backends[plan.backend]
            container_model_path = "/cache/" + str(
                Path(local_path).resolve().relative_to(self._models_dir.resolve())
            )
            # Rebuild plan with the placement-chosen GPU set AND a per-deployment
            # gpu_memory_utilization derived from our reservation. Without this
            # override every vLLM container takes its requested fraction of the
            # *whole* GPU, so two co-located deployments fight for memory and
            # the later one OOMs. We compute fraction = (our share per GPU)
            # / (per-GPU total), with a small safety margin under 1.0.
            tp = len(gpu_ids)
            per_gpu_mb = self._topology.gpus[gpu_ids[0]].total_mb
            per_gpu_reserved = vram_mb / tp
            mem_util = min(0.95, max(0.05, per_gpu_reserved / per_gpu_mb))
            effective_plan = replace(
                plan,
                gpu_ids=list(gpu_ids),
                tensor_parallel=tp,
                gpu_memory_utilization=mem_util,
            )
            handle = self._docker.run(
                image=plan.image_tag,
                name=f"serve-{plan.backend}-{plan.model_name}-{dep.id}",
                command=backend.build_argv(effective_plan, local_model_path=container_model_path),
                environment=backend.container_env(effective_plan),
                kwargs=backend.container_kwargs(effective_plan),
                volumes={str(self._models_dir.resolve()): {"bind": "/cache", "mode": "ro"}},
                internal_port=8000,
            )
            dep_store.set_container(
                self._conn, dep.id,
                container_id=handle.id,
                container_name=handle.name,
                container_port=handle.port,
                container_address=handle.address,
            )

            health_url = f"http://{handle.address}:{handle.port}{backend.health_path}"
            ok = await wait_healthy(health_url, timeout_s=self._load_timeout_s)
            if not ok:
                self._docker.stop(handle.id, timeout=10)
                msg = f"engine did not become healthy within load timeout ({health_url})"
                dep_store.update_status(self._conn, dep.id, "failed", last_error=msg)
                raise RuntimeError(msg)

            dep_store.update_status(self._conn, dep.id, "ready")
            return dep_store.get_by_id(self._conn, dep.id)

    async def stop(self, dep_id: int | None = None) -> None:
        async with self._lock:
            if dep_id is None:
                for d in dep_store.list_ready(self._conn):
                    await self._stop_locked(d.id)
            else:
                await self._stop_locked(dep_id)

    async def pin(self, dep_id: int, pinned: bool = True) -> None:
        dep_store.set_pinned(self._conn, dep_id, pinned)

    async def _stop_locked(self, dep_id: int) -> None:
        dep = dep_store.get_by_id(self._conn, dep_id)
        if dep is None:
            return
        dep_store.update_status(self._conn, dep.id, "stopping")
        if dep.container_id:
            self._docker.stop(dep.container_id, timeout=30)
        dep_store.update_status(self._conn, dep.id, "stopped")
