from __future__ import annotations

import sqlite3
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from serve_engine.backends.base import Backend
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store

router = APIRouter(prefix="/admin")


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
    backend: str = "vllm"
    image_tag: str | None = None
    gpu_ids: list[int]
    tensor_parallel: int | None = None
    max_model_len: int = 8192
    dtype: str = "auto"


class CreateModelRequest(BaseModel):
    name: str
    hf_repo: str
    revision: str = "main"


@router.get("/deployments")
def list_deployments(conn: sqlite3.Connection = Depends(get_conn)):
    return [
        {**asdict(d), "gpu_ids": d.gpu_ids}
        for d in dep_store.list_all(conn)
    ]


@router.post("/deployments", status_code=status.HTTP_201_CREATED)
async def create_deployment(
    body: CreateDeploymentRequest,
    manager: LifecycleManager = Depends(get_manager),
    backends: dict[str, Backend] = Depends(get_backends),
):
    if body.backend not in backends:
        raise HTTPException(400, f"backend {body.backend!r} not supported")
    backend = backends[body.backend]
    image_tag = body.image_tag or backend.image_default
    tp = body.tensor_parallel or len(body.gpu_ids)
    try:
        plan = DeploymentPlan(
            model_name=body.model_name,
            hf_repo=body.hf_repo,
            revision=body.revision,
            backend=body.backend,
            image_tag=image_tag,
            gpu_ids=body.gpu_ids,
            tensor_parallel=tp,
            max_model_len=body.max_model_len,
            dtype=body.dtype,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    dep = await manager.load(plan)
    return {**asdict(dep), "gpu_ids": dep.gpu_ids}


@router.delete("/deployments/current", status_code=status.HTTP_204_NO_CONTENT)
async def stop_current(manager: LifecycleManager = Depends(get_manager)):
    await manager.stop()


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


@router.delete("/models/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(name: str, conn: sqlite3.Connection = Depends(get_conn)):
    m = model_store.get_by_name(conn, name)
    if m is None:
        raise HTTPException(404, f"model {name!r} not found")
    model_store.delete(conn, m.id)
