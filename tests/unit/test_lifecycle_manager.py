import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from serve_engine.backends.vllm import VLLMBackend
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.lifecycle.topology import GPUInfo, Topology
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


@pytest.fixture
def topo_one_gpu():
    return Topology(
        gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024)],
        _islands={0: frozenset({0})},
    )


def _patch_externals(monkeypatch, tmp_path, vram_mb=20_000):
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.estimate_vram_mb",
        lambda inp: vram_mb,
    )
    (tmp_path / "weights").mkdir(exist_ok=True)


def test_load_starts_engine_and_marks_ready(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="vllm-llama-1b", address="127.0.0.1", port=49152,
    )
    _patch_externals(monkeypatch, tmp_path, vram_mb=20_000)

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    dep = asyncio.run(mgr.load(_make_plan()))
    assert dep.status == "ready"
    assert dep.container_id == "cid"
    assert dep.container_address == "127.0.0.1"
    assert dep.vram_reserved_mb == 20_000
    docker_client.run.assert_called_once()


def test_load_evicts_previous_when_room_constrained(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.side_effect = [
        ContainerHandle(id="cid1", name="x1", address="127.0.0.1", port=49152),
        ContainerHandle(id="cid2", name="x2", address="127.0.0.1", port=49153),
    ]
    _patch_externals(monkeypatch, tmp_path, vram_mb=60 * 1024)

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(_make_plan()))
    asyncio.run(mgr.load(_make_plan()))
    # First container must have been stopped to make room
    docker_client.stop.assert_called_once_with("cid1", timeout=30)


def test_pin_prevents_eviction(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid1", name="x1", address="127.0.0.1", port=49152,
    )
    _patch_externals(monkeypatch, tmp_path, vram_mb=60 * 1024)

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    plan_pinned = replace(_make_plan(), pinned=True)
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(plan_pinned))
    # Loading another deployment that needs the same room must fail
    plan2 = _make_plan()
    with pytest.raises(RuntimeError, match="cannot place"):
        asyncio.run(mgr.load(plan2))


def test_load_marks_failed_on_unhealthy(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="127.0.0.1", port=49152,
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.estimate_vram_mb",
        lambda inp: 20_000,
    )
    (tmp_path / "w").mkdir(exist_ok=True)

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    with pytest.raises(RuntimeError, match="did not become healthy"):
        asyncio.run(mgr.load(_make_plan()))
    docker_client.stop.assert_called_once()
    # Failed-load containers MUST be preserved (remove=False) so the engine
    # logs survive for `docker logs` inspection — otherwise root cause is lost.
    stop_kwargs = docker_client.stop.call_args.kwargs
    assert stop_kwargs.get("remove") is False, stop_kwargs
    assert dep_store.find_active(conn) is None


def test_reconcile_marks_orphan_failed(conn, monkeypatch, tmp_path, topo_one_gpu):
    """A ready row whose container no longer exists must be marked failed."""
    from docker.errors import NotFound
    docker_client = MagicMock()

    # Make containers.get raise NotFound for our orphan
    inner_client = MagicMock()
    inner_client.containers.get.side_effect = NotFound("gone")
    docker_client._client = inner_client

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    # Seed a fake ready deployment in the DB
    m = model_store.add(conn, name="llama-1b", hf_repo="org/x")
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=4096, dtype="auto",
    )
    dep_store.set_container(
        conn, d.id,
        container_id="orphan_cid", container_name="x",
        container_port=49152, container_address="127.0.0.1",
    )
    dep_store.update_status(conn, d.id, "ready")

    asyncio.run(mgr.reconcile())

    refreshed = dep_store.get_by_id(conn, d.id)
    assert refreshed.status == "failed"
    assert "disappeared" in (refreshed.last_error or "")


def test_reconcile_keeps_running_container(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    inner_client = MagicMock()
    fake_container = MagicMock()
    fake_container.status = "running"
    inner_client.containers.get.return_value = fake_container
    docker_client._client = inner_client

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    m = model_store.add(conn, name="llama-1b", hf_repo="org/x")
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=4096, dtype="auto",
    )
    dep_store.set_container(
        conn, d.id,
        container_id="live_cid", container_name="x",
        container_port=49152, container_address="127.0.0.1",
    )
    dep_store.update_status(conn, d.id, "ready")

    asyncio.run(mgr.reconcile())

    refreshed = dep_store.get_by_id(conn, d.id)
    assert refreshed.status == "ready"  # unchanged


def test_stop_all_stops_every_ready(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    m = model_store.add(conn, name="llama-1b", hf_repo="org/x")
    for cid in ("c1", "c2", "c3"):
        d = dep_store.create(
            conn, model_id=m.id, backend="vllm", image_tag="img:v1",
            gpu_ids=[0], tensor_parallel=1, max_model_len=4096, dtype="auto",
        )
        dep_store.set_container(
            conn, d.id, container_id=cid, container_name=cid,
            container_port=49152, container_address="127.0.0.1",
        )
        dep_store.update_status(conn, d.id, "ready")

    asyncio.run(mgr.stop_all())

    # All three should be stopped
    statuses = [dep.status for dep in dep_store.list_all(conn)]
    assert all(s == "stopped" for s in statuses)
    assert docker_client.stop.call_count == 3
