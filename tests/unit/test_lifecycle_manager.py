import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from serve_engine.backends.vllm import VLLMBackend
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.store import db
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store


def _make_plan() -> DeploymentPlan:
    return DeploymentPlan(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="vllm",
        image_tag="vllm/vllm-openai:v0.7.3",
        gpu_ids=[0],
        max_model_len=8192,
    )


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    return c


def test_load_starts_engine_and_marks_ready(conn, monkeypatch, tmp_path):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="vllm-llama-1b", address="vllm-llama-1b", port=8000
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )

    mgr = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )

    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    dep = asyncio.run(mgr.load(_make_plan()))
    assert dep.status == "ready"
    assert dep.container_id == "cid"
    docker_client.run.assert_called_once()


def test_load_stops_previous_deployment(conn, monkeypatch, tmp_path):
    docker_client = MagicMock()
    docker_client.run.side_effect = [
        ContainerHandle(id="cid1", name="x1", address="x1", port=8000),
        ContainerHandle(id="cid2", name="x2", address="x2", port=8000),
    ]
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )

    mgr = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(_make_plan()))
    asyncio.run(mgr.load(_make_plan()))
    docker_client.stop.assert_called_once_with("cid1", timeout=30)


def test_load_marks_failed_on_unhealthy(conn, monkeypatch, tmp_path):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(id="cid", name="x", address="x", port=8000)
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )
    mgr = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    with pytest.raises(RuntimeError, match="did not become healthy"):
        asyncio.run(mgr.load(_make_plan()))
    docker_client.stop.assert_called_once()
    assert dep_store.find_active(conn) is None
