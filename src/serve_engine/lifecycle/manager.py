from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import httpx

from serve_engine.backends.base import Backend
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.lifecycle.downloader import download_model
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store

log = logging.getLogger(__name__)


async def download_model_async(**kwargs) -> str:
    # snapshot_download is blocking; offload to a thread
    return await asyncio.to_thread(download_model, **kwargs)


async def wait_healthy(url: str, *, timeout_s: float = 600.0, interval_s: float = 2.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while asyncio.get_event_loop().time() < deadline:
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
        load_timeout_s: float = 600.0,
    ):
        self._conn = conn
        self._docker = docker_client
        self._backends = backends
        self._models_dir = models_dir
        self._load_timeout_s = load_timeout_s
        self._lock = asyncio.Lock()

    @property
    def active(self):
        return dep_store.find_active(self._conn)

    async def load(self, plan: DeploymentPlan):
        async with self._lock:
            model = model_store.get_by_name(self._conn, plan.model_name)
            if model is None:
                model = model_store.add(
                    self._conn, name=plan.model_name, hf_repo=plan.hf_repo, revision=plan.revision
                )

            current = dep_store.find_active(self._conn)
            if current is not None:
                await self._stop_locked(current.id)

            local_path = model.local_path
            if local_path is None:
                local_path = await download_model_async(
                    hf_repo=plan.hf_repo,
                    revision=plan.revision,
                    cache_dir=self._models_dir,
                )
                model_store.set_local_path(self._conn, model.id, local_path)

            dep = dep_store.create(
                self._conn,
                model_id=model.id,
                backend=plan.backend,
                image_tag=plan.image_tag,
                gpu_ids=plan.gpu_ids,
                tensor_parallel=plan.tensor_parallel,
                max_model_len=plan.max_model_len,
                dtype=plan.dtype,
            )
            dep_store.update_status(self._conn, dep.id, "loading")

            backend = self._backends[plan.backend]
            handle = self._docker.run(
                image=plan.image_tag,
                name=f"serve-{plan.backend}-{plan.model_name}-{dep.id}",
                command=backend.build_argv(plan, local_model_path="/model"),
                environment=backend.container_env(plan),
                kwargs=backend.container_kwargs(plan),
                volumes={local_path: {"bind": "/model", "mode": "ro"}},
                internal_port=8000,
            )
            dep_store.set_container(
                self._conn,
                dep.id,
                container_id=handle.id,
                container_name=handle.name,
                container_port=handle.port,
            )

            health_url = f"http://{handle.address}:{handle.port}{backend.health_path}"
            ok = await wait_healthy(health_url, timeout_s=self._load_timeout_s)
            if not ok:
                self._docker.stop(handle.id, timeout=10)
                dep_store.update_status(
                    self._conn, dep.id, "failed",
                    last_error="engine did not become healthy within load timeout",
                )
                raise RuntimeError("engine did not become healthy within load timeout")

            dep_store.update_status(self._conn, dep.id, "ready")
            return dep_store.get_by_id(self._conn, dep.id)

    async def stop(self) -> None:
        async with self._lock:
            current = dep_store.find_active(self._conn)
            if current is None:
                return
            await self._stop_locked(current.id)

    async def _stop_locked(self, dep_id: int) -> None:
        dep = dep_store.get_by_id(self._conn, dep_id)
        if dep is None:
            return
        dep_store.update_status(self._conn, dep.id, "stopping")
        if dep.container_id:
            self._docker.stop(dep.container_id, timeout=30)
        dep_store.update_status(self._conn, dep.id, "stopped")
