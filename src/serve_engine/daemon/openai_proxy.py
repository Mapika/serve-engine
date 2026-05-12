from __future__ import annotations

import json
import sqlite3

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from serve_engine.auth.middleware import require_auth_dep
from serve_engine.backends.base import Backend
from serve_engine.lifecycle.adapter_router import (
    UnknownModel,
    ensure_adapter_loaded,
    find_deployment_for,
    resolve_target,
)
from serve_engine.store import adapters as ad_store
from serve_engine.store import api_keys as _api_keys_store
from serve_engine.store import deployment_adapters as da_store
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

    # Resolve `model` to (base, optional adapter). Bare base names route
    # exactly as v1 did. Adapter names cause us to (a) pick a deployment
    # of the adapter's base that has the adapter loaded or can hot-load
    # it, and (b) rewrite the upstream payload's `model` to the adapter
    # name so vLLM/SGLang dispatch against the right LoRA slot.
    try:
        target = resolve_target(conn, model_name)
    except UnknownModel as e:
        raise HTTPException(404, detail=str(e)) from e

    active = find_deployment_for(conn, target.base_model_name, target.adapter_name)
    if active is None or active.container_address is None:
        # Differentiate: no deployment at all vs adapter requested but no
        # LoRA-enabled deployment available.
        if target.adapter_name:
            detail = (
                f"no ready deployment of base {target.base_model_name!r} "
                f"with --max-loras > 0 for adapter {target.adapter_name!r}"
            )
        else:
            detail = f"no ready deployment for model {model_name!r}"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

    backend = backends.get(active.backend)
    if backend is None:
        raise HTTPException(500, detail=f"unknown backend {active.backend!r}")

    # If adapter requested, ensure it's loaded into the chosen deployment.
    # This is the hot-load path: ~100-500ms for the first request to a
    # given adapter; sub-second on subsequent requests (already loaded).
    if target.adapter_name:
        manager = request.app.state.manager
        try:
            await ensure_adapter_loaded(
                conn, backend, active, target.adapter_name,
                models_dir=manager._models_dir,
            )
        except UnknownModel as e:
            raise HTTPException(404, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(502, detail=f"adapter load failed: {e}") from e

    dep_store.touch_last_request(conn, active.id)
    request.app.state.request_count += 1

    # Rewrite the upstream payload's `model` field to the adapter name
    # when an adapter is in play — vLLM/SGLang both treat the OpenAI
    # `model` field as the LoRA slot name when --enable-lora is on.
    if target.adapter_name and target.adapter_name != model_name:
        try:
            parsed["model"] = target.adapter_name
            body = json.dumps(parsed).encode()
        except (TypeError, json.JSONDecodeError):
            pass  # body already validated as JSON above; should not happen
    elif target.adapter_name:
        # bare adapter name was passed; no rewrite needed
        pass

    base = f"http://{active.container_address}:{active.container_port}{backend.openai_base}"
    _HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    # Strip the user's Authorization (don't leak the API key to the engine).
    headers.pop("authorization", None)
    headers.pop("Authorization", None)

    # Open the upstream stream BEFORE returning so we can read its status
    # and headers and forward them faithfully to the caller.
    client = make_engine_client(base)
    stream_cm = client.stream("POST", openai_subpath, content=body, headers=headers)
    resp = await stream_cm.__aenter__()

    # Forward upstream status + content-type + selected headers.
    upstream_ct = resp.headers.get("content-type", "application/octet-stream")
    forward_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"content-type"}
    }

    # Track only the last SSE event's `data:` payload, or the entire body if
    # it's a single JSON object. ~4 KB cap on the last-line buffer is plenty
    # for usage payloads which are tens of bytes.
    usage_tracker = _UsageTracker(is_sse="event-stream" in upstream_ct)

    async def streamer():
        try:
            async for chunk in resp.aiter_raw():
                usage_tracker.feed(chunk)
                yield chunk
        finally:
            await stream_cm.__aexit__(None, None, None)
            await client.aclose()
            if key is not None:
                tin, tout = usage_tracker.extract()
                _key_usage_store.record(
                    conn, key_id=key.id, tokens_in=tin, tokens_out=tout,
                    model_name=model_name,
                )

    return StreamingResponse(
        streamer(),
        status_code=resp.status_code,
        headers=forward_headers,
        media_type=upstream_ct,
    )


class _UsageTracker:
    """Best-effort token-count extraction without buffering the full response.

    - SSE mode: keep the last complete `data: {...}` event; OpenAI/vLLM/SGLang
      emit usage in the final event when stream_options.include_usage=true.
    - JSON mode: buffer up to a small cap (single JSON response). Most non-
      streaming `/v1/chat/completions` bodies are well under 64 KB.
    """

    _MAX_JSON = 65_536

    def __init__(self, *, is_sse: bool):
        self._is_sse = is_sse
        self._last_event = bytearray()
        self._current = bytearray()
        self._json_buf = bytearray()
        self._json_overflow = False

    def feed(self, chunk: bytes) -> None:
        if self._is_sse:
            self._current.extend(chunk)
            # An event ends with a blank line (\n\n). When we see one,
            # the bytes before the blank line are the most recent event.
            while True:
                idx = self._current.find(b"\n\n")
                if idx < 0:
                    break
                event = bytes(self._current[:idx])
                del self._current[: idx + 2]
                if b"data:" in event:
                    self._last_event = bytearray(event)
        else:
            if not self._json_overflow:
                remaining = self._MAX_JSON - len(self._json_buf)
                if remaining <= 0:
                    self._json_overflow = True
                else:
                    self._json_buf.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        self._json_overflow = True

    def extract(self) -> tuple[int, int]:
        if self._is_sse:
            payload = self._last_event
            if not payload:
                return 0, 0
            # Strip lines that don't start with `data:`
            for line in payload.split(b"\n"):
                if line.startswith(b"data:"):
                    body = line[len(b"data:"):].strip()
                    if body == b"[DONE]" or not body:
                        continue
                    try:
                        obj = json.loads(body)
                        u = obj.get("usage") or {}
                        return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
                    except (json.JSONDecodeError, AttributeError):
                        continue
            return 0, 0
        # JSON mode
        if self._json_overflow or not self._json_buf:
            return 0, 0
        try:
            obj = json.loads(bytes(self._json_buf))
            u = obj.get("usage") or {}
            return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        except (json.JSONDecodeError, AttributeError):
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
    base_entries = [
        {
            "id": m.name,
            "object": "model",
            "owned_by": "serve-engine",
            "loaded": m.id in ready_by_model,
            "pinned": ready_by_model[m.id].pinned if m.id in ready_by_model else False,
        }
        for m in rows
    ]
    # Adapters appear alongside base models — clients can `model=<adapter>`
    # directly. `base` field disambiguates for clients that want the parent.
    adapter_entries = [
        {
            "id": a.name,
            "object": "model",
            "owned_by": "serve-engine",
            "base": a.base_model.name,
            "loaded": bool(da_store.find_deployments_with_adapter(conn, a.id)),
            "downloaded": a.local_path is not None,
        }
        for a in ad_store.list_all(conn)
    ]
    return {"object": "list", "data": base_entries + adapter_entries}
