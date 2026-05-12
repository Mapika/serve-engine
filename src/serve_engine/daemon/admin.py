from __future__ import annotations

import asyncio
import json as _json
import sqlite3
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from serve_engine.auth.middleware import require_auth_dep
from serve_engine.backends.base import Backend
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.observability.gpu_stats import read_gpu_stats as _read_gpu_stats
from serve_engine.store import api_keys as _ak_store
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store


def _is_uds_request(request: Request) -> bool:
    """True when the request arrived over the Unix domain socket, not TCP.

    Uvicorn's UDS server reports scope['client'] as None (no remote address)
    whereas TCP delivers a (host, port) tuple. We use 'client' rather than
    'server' because uvicorn fills 'server' with the listening address even
    on UDS (e.g. ('', 0)).
    """
    client = request.scope.get("client")
    return client is None


def require_admin_key(
    request: Request,
) -> _ak_store.ApiKey | None:
    """Authorize /admin/*.

    Trust model:
    - Local UDS requests bypass auth entirely. The user controls the host
      filesystem; presence on the socket is sufficient. This is also the
      bootstrap path: `serve key create web --tier admin` over UDS works
      even after other tier=admin keys exist.
    - TCP requests fall through to require_auth_dep. If no keys exist at
      all, that dep also bypasses (homelab UX). Otherwise it requires a
      valid Bearer; we then further require tier=admin here.
    """
    if _is_uds_request(request):
        return None
    key = require_auth_dep(request)
    if key is None:
        return None
    if key.tier != "admin":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="admin tier required for /admin/*",
        )
    return key


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)])


def get_manager(request: Request) -> LifecycleManager:
    return request.app.state.manager


def get_conn(request: Request) -> sqlite3.Connection:
    return request.app.state.conn


def get_backends(request: Request) -> dict[str, Backend]:
    return request.app.state.backends


class CreateDeploymentRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    hf_repo: str
    revision: str = "main"
    backend: str | None = None   # default → selection rules
    image_tag: str | None = None
    gpu_ids: list[int]
    tensor_parallel: int | None = None
    max_model_len: int = 8192
    dtype: str = "auto"
    pinned: bool = False
    idle_timeout_s: int | None = None
    target_concurrency: int | None = None
    extra_args: dict[str, str] = {}


class CreateModelRequest(BaseModel):
    name: str
    hf_repo: str
    revision: str = "main"


@router.get("/deployments")
def list_deployments(
    conn: sqlite3.Connection = Depends(get_conn),
    manager: LifecycleManager = Depends(get_manager),
):
    # Live per-process VRAM (NVML compute view) joined to each deployment by
    # the union of pids in its docker container. vram_reserved_mb is the
    # estimator's pre-load reservation; vram_used_mb is what nvidia-smi sees.
    from serve_engine.observability.gpu_stats import read_compute_process_vram
    pid_vram = read_compute_process_vram()
    out = []
    for d in dep_store.list_all(conn):
        used_mb: int | None = None
        if pid_vram and d.container_id and d.status in ("loading", "ready"):
            try:
                pids = manager._docker.container_pids(d.container_id)
                total = sum(pid_vram.get(p, 0) for p in pids)
                used_mb = total or None
            except Exception:
                used_mb = None
        out.append({**asdict(d), "gpu_ids": d.gpu_ids, "vram_used_mb": used_mb})
    return out


@router.post("/deployments", status_code=status.HTTP_201_CREATED)
async def create_deployment(
    body: CreateDeploymentRequest,
    manager: LifecycleManager = Depends(get_manager),
    backends: dict[str, Backend] = Depends(get_backends),
):
    from serve_engine.backends.selection import load_selection, pick_backend
    backend_name = body.backend
    if backend_name is None:
        backend_name = pick_backend(load_selection(), body.model_name)
    if backend_name not in backends:
        raise HTTPException(400, f"backend {backend_name!r} not supported")
    backend = backends[backend_name]
    image_tag = body.image_tag or backend.image_default
    tp = body.tensor_parallel or len(body.gpu_ids)
    try:
        plan = DeploymentPlan(
            model_name=body.model_name,
            hf_repo=body.hf_repo,
            revision=body.revision,
            backend=backend_name,
            image_tag=image_tag,
            gpu_ids=body.gpu_ids,
            tensor_parallel=tp,
            max_model_len=body.max_model_len,
            dtype=body.dtype,
            pinned=body.pinned,
            idle_timeout_s=body.idle_timeout_s,
            target_concurrency=body.target_concurrency,
            extra_args=dict(body.extra_args),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    dep = await manager.load(plan)
    return {**asdict(dep), "gpu_ids": dep.gpu_ids}


@router.delete("/deployments/{dep_id}", status_code=status.HTTP_204_NO_CONTENT)
async def stop_deployment(
    dep_id: int,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if dep_store.get_by_id(conn, dep_id) is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    await manager.stop(dep_id)


@router.delete("/deployments", status_code=status.HTTP_204_NO_CONTENT)
async def stop_all_deployments(manager: LifecycleManager = Depends(get_manager)):
    await manager.stop_all()


@router.get("/models")
def list_models(conn: sqlite3.Connection = Depends(get_conn)):
    return [asdict(m) for m in model_store.list_all(conn)]


@router.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(
    body: CreateModelRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    try:
        m = model_store.add(conn, name=body.name, hf_repo=body.hf_repo, revision=body.revision)
    except model_store.AlreadyExists as e:
        raise HTTPException(409, str(e)) from e
    return asdict(m)


@router.post("/models/{name}/download")
async def download_model_endpoint(
    name: str,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Synchronously download a registered model's weights to the local cache.

    Returns when the download completes (or raises 5xx if it fails).
    The CLI's `serve pull` calls this after registering; clients should
    expect potentially many minutes for large models.
    """
    m = model_store.get_by_name(conn, name)
    if m is None:
        raise HTTPException(404, f"model {name!r} not registered")
    if m.local_path is not None:
        return {"name": m.name, "local_path": m.local_path, "already_present": True}
    from serve_engine.lifecycle.downloader import download_model
    try:
        local_path = await asyncio.to_thread(
            download_model,
            hf_repo=m.hf_repo,
            revision=m.revision,
            cache_dir=manager._models_dir,
        )
    except Exception as e:
        raise HTTPException(502, f"download failed: {e}") from e
    model_store.set_local_path(conn, m.id, local_path)
    return {"name": m.name, "local_path": local_path, "already_present": False}


@router.delete("/models/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(name: str, conn: sqlite3.Connection = Depends(get_conn)):
    m = model_store.get_by_name(conn, name)
    if m is None:
        raise HTTPException(404, f"model {name!r} not found")
    model_store.delete(conn, m.id)


@router.post("/deployments/{dep_id}/pin", status_code=status.HTTP_204_NO_CONTENT)
async def pin_deployment(
    dep_id: int,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if dep_store.get_by_id(conn, dep_id) is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    await manager.pin(dep_id, True)


@router.post("/deployments/{dep_id}/unpin", status_code=status.HTTP_204_NO_CONTENT)
async def unpin_deployment(
    dep_id: int,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if dep_store.get_by_id(conn, dep_id) is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    await manager.pin(dep_id, False)


@router.get("/deployments/current/logs")
def stream_current_logs(request: Request):
    conn: sqlite3.Connection = request.app.state.conn
    docker_client = request.app.state.manager._docker  # Plan 02 promotes a public accessor
    active = dep_store.find_active(conn)
    if active is None or active.container_id is None:
        raise HTTPException(404, "no active deployment with a running container")

    def gen():
        for chunk in docker_client.stream_logs(active.container_id, follow=True):
            if isinstance(chunk, bytes):
                yield chunk
            else:
                yield chunk.encode()

    return StreamingResponse(gen(), media_type="text/plain")


@router.get("/deployments/{dep_id}/logs/stream")
async def stream_engine_logs_sse(dep_id: int, request: Request) -> StreamingResponse:
    """SSE: stdout/stderr of the engine container for this deployment.

    Designed for the browser EventSource — each docker log chunk is reframed
    as one or more `data: <line>` SSE events. Includes the last 500 lines as
    history before following. The underlying sync iterator from docker-py is
    bridged into the event loop via asyncio.to_thread; on client disconnect
    the consumer task is cancelled but the docker iterator may continue to
    drain in the background until the container stops (small thread leak,
    acceptable since these are bounded by container lifetime).
    """
    conn: sqlite3.Connection = request.app.state.conn
    docker_client = request.app.state.manager._docker
    dep = dep_store.get_by_id(conn, dep_id)
    if dep is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    if dep.container_id is None:
        raise HTTPException(404, f"deployment {dep_id} has no container")

    async def gen():
        yield ":ok\n\n"
        try:
            sync_iter = docker_client.stream_logs(
                dep.container_id, follow=True, tail=500,
            )
        except Exception as e:
            yield f"data: [serve-engine] failed to attach: {e}\n\n"
            return
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, sync_iter, sentinel)
            if chunk is sentinel:
                yield "data: [serve-engine] log stream ended\n\n"
                return
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8", errors="replace")
            else:
                text = str(chunk)
            for line in text.splitlines():
                # Each SSE event must be a single line; collapse blank lines.
                if line:
                    yield f"data: {line}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


class CreateKeyRequest(BaseModel):
    name: str
    tier: str = "standard"
    rpm_override: int | None = None
    tpm_override: int | None = None
    rph_override: int | None = None
    tph_override: int | None = None
    rpd_override: int | None = None
    tpd_override: int | None = None
    rpw_override: int | None = None
    tpw_override: int | None = None


@router.get("/keys")
def list_keys(conn: sqlite3.Connection = Depends(get_conn)):
    return [
        {
            "id": k.id,
            "name": k.name,
            "prefix": k.prefix,
            "tier": k.tier,
            "revoked": k.revoked_at is not None,
        }
        for k in _ak_store.list_all(conn)
    ]


@router.post("/keys", status_code=status.HTTP_201_CREATED)
def create_key(
    body: CreateKeyRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    secret, k = _ak_store.create(
        conn, name=body.name, tier=body.tier,
        rpm_override=body.rpm_override, tpm_override=body.tpm_override,
        rph_override=body.rph_override, tph_override=body.tph_override,
        rpd_override=body.rpd_override, tpd_override=body.tpd_override,
        rpw_override=body.rpw_override, tpw_override=body.tpw_override,
    )
    return {
        "id": k.id,
        "name": k.name,
        "prefix": k.prefix,
        "tier": k.tier,
        "secret": secret,
    }


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: int,
    conn: sqlite3.Connection = Depends(get_conn),
):
    if _ak_store.get_by_id(conn, key_id) is None:
        raise HTTPException(404, f"no key with id {key_id}")
    _ak_store.revoke(conn, key_id)


@router.get("/events")
async def events(request: Request) -> StreamingResponse:
    """SSE: lifecycle events as `data: <json>\n\n` chunks. Heartbeat every 15s."""
    bus = request.app.state.event_bus

    async def gen():
        async with bus.subscribe() as queue:
            yield ":ok\n\n"  # initial heartbeat
            while True:
                try:
                    e = await asyncio.wait_for(queue.get(), timeout=15.0)
                    payload = _json.dumps({
                        "kind": e.kind, "payload": e.payload, "ts": e.ts,
                    })
                    yield f"data: {payload}\n\n"
                except TimeoutError:
                    yield ":hb\n\n"  # SSE comment heartbeat

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/gpus")
def list_gpus():
    """Per-GPU live snapshot: memory, utilization, power."""
    return [
        {
            "index": s.index,
            "memory_used_mb": s.memory_used_mb,
            "memory_total_mb": s.memory_total_mb,
            "gpu_util_pct": s.gpu_util_pct,
            "power_w": s.power_w,
        }
        for s in _read_gpu_stats()
    ]
