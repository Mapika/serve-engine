from __future__ import annotations

import sqlite3

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from serve_engine.backends.base import Backend
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store

router = APIRouter()

ENGINE_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)


def make_engine_client(base_url: str) -> httpx.AsyncClient:
    """Factory wrapper so tests can monkeypatch transport."""
    return httpx.AsyncClient(base_url=base_url, timeout=ENGINE_TIMEOUT)


async def _proxy(request: Request, openai_subpath: str) -> StreamingResponse:
    conn: sqlite3.Connection = request.app.state.conn
    backends: dict[str, Backend] = request.app.state.backends

    active = dep_store.find_active(conn)
    if active is None or active.status != "ready":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no active deployment ready to serve",
        )

    backend = backends.get(active.backend)
    if backend is None:
        raise HTTPException(500, detail=f"unknown backend {active.backend!r}")

    # Engine containers publish their port on 127.0.0.1 via the C1 fix.
    # Plan 02 should add a container_address column for clarity.
    base = f"http://127.0.0.1:{active.container_port}{backend.openai_base}"
    body = await request.body()
    _HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }

    client = make_engine_client(base)
    upstream = client.stream("POST", openai_subpath, content=body, headers=headers)

    async def streamer():
        try:
            async with upstream as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(streamer(), media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy(request, "/chat/completions")


@router.post("/v1/completions")
async def completions(request: Request):
    return await _proxy(request, "/completions")


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _proxy(request, "/embeddings")


@router.get("/v1/models")
def models(request: Request):
    conn: sqlite3.Connection = request.app.state.conn
    active = dep_store.find_active(conn)
    rows = model_store.list_all(conn)
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "owned_by": "serve-engine",
                "loaded": (
                    active is not None
                    and active.model_id == m.id
                    and active.status == "ready"
                ),
            }
            for m in rows
        ],
    }
