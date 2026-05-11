from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from serve_engine.backends.vllm import VLLMBackend
from serve_engine.daemon.app import build_app
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.store import db


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="x", port=8000
    )
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_list_deployments_empty(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/admin/deployments")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_create_deployment(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post(
            "/admin/deployments",
            json={
                "model_name": "llama-1b",
                "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
                "image_tag": "vllm/vllm-openai:v0.7.3",
                "gpu_ids": [0],
                "max_model_len": 8192,
            },
        )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "ready"


@pytest.mark.asyncio
async def test_list_models(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post(
            "/admin/models",
            json={"name": "x", "hf_repo": "org/x"},
        )
        r = await c.get("/admin/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "x" in names
