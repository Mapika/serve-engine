import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from serve_engine.backends.vllm import VLLMBackend
from serve_engine.daemon.app import build_app
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.store import db


class FakeEngineApp:
    """A tiny ASGI app pretending to be the upstream engine."""

    def __init__(self, response_chunks: list[bytes], status_code: int = 200):
        self.chunks = response_chunks
        self.status_code = status_code
        self.last_request_body: bytes | None = None

    async def __call__(self, scope, receive, send):
        assert scope["type"] == "http"
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body"):
                break
        self.last_request_body = body
        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": [(b"content-type", b"text/event-stream")],
        })
        for i, chunk in enumerate(self.chunks):
            await send({
                "type": "http.response.body",
                "body": chunk,
                "more_body": i < len(self.chunks) - 1,
            })


@pytest.fixture
def app_with_active_deployment(tmp_path, monkeypatch):
    fake_engine = FakeEngineApp([b"data: hello\n\n", b"data: [DONE]\n\n"])

    def fake_async_client_factory(base_url):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fake_engine),
            base_url="http://engine",
        )
    monkeypatch.setattr(
        "serve_engine.daemon.openai_proxy.make_engine_client",
        fake_async_client_factory,
    )

    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="engine", address="engine", port=8000
    )
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    app = build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )

    async def setup():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30
        ) as c:
            r = await c.post("/admin/deployments", json={
                "model_name": "llama-1b",
                "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
                "image_tag": "img:v1",
                "gpu_ids": [0],
                "max_model_len": 8192,
            })
            assert r.status_code == 201
    asyncio.run(setup())
    return app, fake_engine


@pytest.mark.asyncio
async def test_proxy_streams_response(app_with_active_deployment):
    app, fake = app_with_active_deployment
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30
    ) as c:
        async with c.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "llama-1b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            chunks = [c async for c in r.aiter_bytes()]
    assert r.status_code == 200
    body = b"".join(chunks)
    assert b"hello" in body
    assert b"[DONE]" in body
    forwarded = json.loads(fake.last_request_body)
    assert forwarded["model"] == "llama-1b"


@pytest.mark.asyncio
async def test_proxy_404_when_no_active(tmp_path, monkeypatch):
    docker_client = MagicMock()
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    app = build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "llama-1b", "messages": []},
        )
    assert r.status_code == 503
    assert "no active deployment" in r.json()["detail"]
