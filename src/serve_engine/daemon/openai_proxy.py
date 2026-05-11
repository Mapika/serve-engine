from __future__ import annotations

import json
import sqlite3

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from serve_engine.auth.middleware import require_auth_dep
from serve_engine.backends.base import Backend
from serve_engine.store import api_keys as _api_keys_store
from serve_engine.store import deployments as dep_store
from serve_engine.store import key_usage as _key_usage_store
from serve_engine.store import models as model_store

router = APIRouter()

ENGINE_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)


def make_engine_client(base_url: str) -> httpx.AsyncClient:
    """Factory wrapper so tests can monkeypatch transport."""
    return httpx.AsyncClient(base_url=base_url, timeout=ENGINE_TIMEOUT)


async def _proxy(
    request: Request,
    openai_subpath: str,
    *,
    key: _api_keys_store.ApiKey | None,
) -> StreamingResponse:
    conn: sqlite3.Connection = request.app.state.conn
    backends: dict[str, Backend] = request.app.state.backends

    body = await request.body()
    model_name: str | None = None
    try:
        parsed = json.loads(body) if body else {}
        if isinstance(parsed, dict):
            model_name = parsed.get("model")
    except json.JSONDecodeError:
        pass

    if not model_name:
        raise HTTPException(400, detail="request body must include 'model'")

    active = dep_store.find_ready_by_model_name(conn, model_name)
    if active is None or active.container_address is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"no ready deployment for model {model_name!r}",
        )

    backend = backends.get(active.backend)
    if backend is None:
        raise HTTPException(500, detail=f"unknown backend {active.backend!r}")

    dep_store.touch_last_request(conn, active.id)
    request.app.state.request_count = getattr(request.app.state, "request_count", 0) + 1

    base = f"http://{active.container_address}:{active.container_port}{backend.openai_base}"
    _HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    # Strip the user's Authorization (don't leak the API key to the engine).
    headers.pop("authorization", None)
    headers.pop("Authorization", None)

    client = make_engine_client(base)
    upstream = client.stream("POST", openai_subpath, content=body, headers=headers)

    captured = bytearray()

    async def streamer():
        try:
            async with upstream as resp:
                async for chunk in resp.aiter_raw():
                    captured.extend(chunk)
                    yield chunk
        finally:
            await client.aclose()
            if key is not None:
                tin, tout = _extract_usage(bytes(captured))
                _key_usage_store.record(
                    conn, key_id=key.id, tokens_in=tin, tokens_out=tout,
                    model_name=model_name,
                )

    return StreamingResponse(streamer(), media_type="text/event-stream")


def _extract_usage(body: bytes) -> tuple[int, int]:
    """Best-effort token-count extraction from OpenAI-format response or SSE."""
    if not body:
        return 0, 0
    text = body.decode(errors="replace")
    # Try non-streaming JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            usage = obj.get("usage") or {}
            return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    except json.JSONDecodeError:
        pass
    # SSE: vLLM/SGLang emit a final `data: { ... "usage": {...} ...}` chunk for streams
    # that opt in via stream_options.include_usage=true.
    for line in reversed(text.splitlines()):
        if line.startswith("data:") and "usage" in line:
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
                usage = obj.get("usage") or {}
                return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
            except json.JSONDecodeError:
                continue
    return 0, 0


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    key: _api_keys_store.ApiKey | None = Depends(require_auth_dep),
):
    return await _proxy(request, "/chat/completions", key=key)


@router.post("/v1/completions")
async def completions(
    request: Request,
    key: _api_keys_store.ApiKey | None = Depends(require_auth_dep),
):
    return await _proxy(request, "/completions", key=key)


@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    key: _api_keys_store.ApiKey | None = Depends(require_auth_dep),
):
    return await _proxy(request, "/embeddings", key=key)


@router.get("/v1/models")
def models(
    request: Request,
    _key: _api_keys_store.ApiKey | None = Depends(require_auth_dep),
):
    conn: sqlite3.Connection = request.app.state.conn
    ready_by_model: dict[int, dep_store.Deployment] = {}
    for d in dep_store.list_ready(conn):
        ready_by_model[d.model_id] = d
    rows = model_store.list_all(conn)
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "owned_by": "serve-engine",
                "loaded": m.id in ready_by_model,
                "pinned": ready_by_model[m.id].pinned if m.id in ready_by_model else False,
            }
            for m in rows
        ],
    }
