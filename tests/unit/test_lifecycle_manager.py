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


@pytest.fixture
def topo_one_gpu_with_cap():
    return Topology(
        gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024, compute_cap="9.0")],
        _islands={0: frozenset({0})},
    )


def test_load_uses_snapshot_path_when_supported(
    conn, monkeypatch, tmp_path, topo_one_gpu_with_cap,
):
    """When the backend opts into snapshots, manager.load (a) bind-mounts
    ~/.serve/snapshots/<key>/ at /snapshots, (b) injects
    TORCHINDUCTOR_CACHE_DIR=/snapshots/torch_cache, and (c) inserts a
    snapshots row in the background after warmup. Save delay is 0 in
    tests so we can drain pending saves explicitly."""
    captured = {}

    def _capture(**kw):
        captured.update(kw)
        return ContainerHandle(id="cid", name="x", address="127.0.0.1", port=49152)

    docker_client = MagicMock()
    docker_client.run.side_effect = _capture
    _patch_externals(monkeypatch, tmp_path, vram_mb=20_000)

    snapshots_dir = tmp_path / "snapshots"
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu_with_cap,
        snapshots_dir=snapshots_dir,
        snapshot_save_delay_s=0.0,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")

    async def _do():
        dep = await mgr.load(_make_plan())
        await mgr.await_pending_snapshot_saves()
        return dep

    dep = asyncio.run(_do())
    assert dep.status == "ready"

    # Snapshot bind-mount present in container volumes.
    volumes = captured["volumes"]
    snapshot_hosts = [p for p in volumes if "snapshots" in p]
    assert len(snapshot_hosts) == 1
    assert volumes[snapshot_hosts[0]] == {"bind": "/snapshots", "mode": "rw"}

    # vLLM cache root set in container env (vLLM 0.20.2 stores its compile
    # cache under VLLM_CACHE_ROOT, not TORCHINDUCTOR_CACHE_DIR).
    env = captured["environment"]
    assert env.get("VLLM_CACHE_ROOT") == "/snapshots"

    # Snapshot row inserted with size sampled from disk.
    from serve_engine.store import snapshots as snap_store
    rows = snap_store.list_all(conn)
    assert len(rows) == 1
    assert rows[0].engine == "vllm"
    assert rows[0].gpu_arch == "9.0"
    assert rows[0].hf_repo == "meta-llama/Llama-3.2-1B-Instruct"
    assert rows[0].local_path.endswith(rows[0].key)


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


def test_load_stops_prior_deployment_of_same_name(conn, monkeypatch, tmp_path, topo_one_gpu):
    """`serve run X` must stop any existing ready deployment named X before
    starting the new one — that's the CLI contract ("Stops the current
    model first"). Without this, two co-located deployments of the same
    base name burn extra VRAM and the proxy's find_ready_by_model_name
    routes to whichever has the newer started_at, masking config drift
    between the two engine processes.
    """
    docker_client = MagicMock()
    docker_client.run.side_effect = [
        ContainerHandle(id="cid1", name="x1", address="127.0.0.1", port=49152),
        ContainerHandle(id="cid2", name="x2", address="127.0.0.1", port=49153),
    ]
    # Small VRAM per deployment so both would fit on the 80 GB GPU under the
    # current placement logic — placement will NOT evict on its own here.
    _patch_externals(monkeypatch, tmp_path, vram_mb=10_000)

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    dep1 = asyncio.run(mgr.load(_make_plan()))
    dep2 = asyncio.run(mgr.load(_make_plan()))

    # Invariant: at most one ready deployment per model_name. dep1 was
    # superseded by dep2 and must be in 'stopped' status with its container
    # torn down.
    assert dep_store.get_by_id(conn, dep1.id).status == "stopped"
    assert dep_store.get_by_id(conn, dep2.id).status == "ready"
    docker_client.stop.assert_any_call("cid1", timeout=30)


def test_pin_prevents_eviction(conn, monkeypatch, tmp_path, topo_one_gpu):
    """Pinned deployments are not LRU-evicted to make room for a different
    model. The second load is for a *different* model so the same-name
    replacement path doesn't apply — this test is purely about placement.
    """
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
    # Different model that won't fit alongside the pinned one and can't
    # evict it.
    model_store.add(conn, name="other-model", hf_repo="org/other")
    plan2 = replace(_make_plan(), model_name="other-model", hf_repo="org/other")
    with pytest.raises(RuntimeError, match="cannot place"):
        asyncio.run(mgr.load(plan2))


def test_load_refuses_to_replace_pinned_same_name(conn, monkeypatch, tmp_path, topo_one_gpu):
    """`serve run X` on a pinned X errors with a clear message instead of
    silently replacing — pin is the operator's commitment that the
    deployment is special. They must `serve unpin X` first.
    """
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid1", name="x1", address="127.0.0.1", port=49152,
    )
    _patch_externals(monkeypatch, tmp_path, vram_mb=10_000)

    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(replace(_make_plan(), pinned=True)))

    with pytest.raises(RuntimeError, match="is pinned; run `serve unpin"):
        asyncio.run(mgr.load(_make_plan()))
    # Critical: the pinned deployment is still ready and intact.
    docker_client.stop.assert_not_called()


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



def test_load_writes_engine_config_yaml_and_mounts_it(conn, monkeypatch, tmp_path, topo_one_gpu):
    """Backends that return engine_config() get a per-deployment YAML written
    to configs_dir, mounted at /serve/configs:ro, and --config <in-container path>
    threaded through build_argv. TRT-LLM is the consumer; this test uses a
    minimal stub backend so we can assert without pulling the trtllm image."""
    import yaml

    from serve_engine.backends.base import ContainerBackend

    class _StubBackendWithConfig(ContainerBackend):
        name = "stubcfg"

        def __init__(self):
            from serve_engine.backends.manifest import EngineManifest, Headroom
            self.manifest = EngineManifest(
                name="stubcfg", image="stub", pinned_tag="v1",
                health_path="/health", openai_base="/v1",
                metrics_path="/metrics", internal_port=8000,
                headroom=Headroom(factor=1.5, min_extra_mb=2048, min_floor_pct=15),
            )

        def build_argv(self, plan, *, local_model_path, config_path=None):
            argv = ["serve", local_model_path]
            if config_path is not None:
                argv.extend(["--config", config_path])
            return argv

        def engine_config(self, plan):
            return {"hello": "world", "n": plan.target_concurrency}

    # Allow the stub backend through DeploymentPlan validation.
    import serve_engine.lifecycle.plan as plan_mod
    monkeypatch.setattr(plan_mod, "SUPPORTED_BACKENDS", ("vllm", "sglang", "trtllm", "stubcfg"))

    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="stub", address="127.0.0.1", port=49152,
    )
    _patch_externals(monkeypatch, tmp_path, vram_mb=20_000)

    configs_dir = tmp_path / "configs"
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"stubcfg": _StubBackendWithConfig()},
        models_dir=tmp_path, topology=topo_one_gpu,
        configs_dir=configs_dir,
    )
    model_store.add(conn, name="m", hf_repo="x/y")
    plan = replace(_make_plan(), model_name="m", hf_repo="x/y", backend="stubcfg")
    dep = asyncio.run(mgr.load(plan))

    # Config file written with the backend's dict serialized as YAML.
    written = configs_dir / f"{dep.id}.yml"
    assert written.exists(), f"no config at {written}"
    payload = yaml.safe_load(written.read_text())
    assert payload["hello"] == "world"
    # `n` came from plan.target_concurrency. With no config.json on the stub
    # weights dir, default_target_concurrency falls back to floor=8.
    assert payload["n"] == 8

    # Manager mounted configs_dir at /serve/configs:ro and passed the in-container
    # path to build_argv via --config.
    call = docker_client.run.call_args
    volumes = call.kwargs["volumes"]
    assert str(configs_dir.resolve()) in volumes
    assert volumes[str(configs_dir.resolve())] == {"bind": "/serve/configs", "mode": "ro"}
    cmd = call.kwargs["command"]
    i = cmd.index("--config")
    assert cmd[i + 1] == f"/serve/configs/{dep.id}.yml"


def test_stop_removes_engine_config_yaml(conn, monkeypatch, tmp_path, topo_one_gpu):
    """Per-deployment config files are cleaned up on stop so configs_dir
    doesn't accumulate stale files across many load/stop cycles."""
    from serve_engine.backends.base import ContainerBackend

    class _StubBackend(ContainerBackend):
        name = "stubcfg2"

        def __init__(self):
            from serve_engine.backends.manifest import EngineManifest, Headroom
            self.manifest = EngineManifest(
                name="stubcfg2", image="stub", pinned_tag="v1",
                health_path="/health", openai_base="/v1",
                metrics_path="/metrics", internal_port=8000,
                headroom=Headroom(factor=1.5, min_extra_mb=2048, min_floor_pct=15),
            )

        def build_argv(self, plan, *, local_model_path, config_path=None):
            return ["serve", local_model_path]

        def engine_config(self, plan):
            return {"k": "v"}

    import serve_engine.lifecycle.plan as plan_mod
    monkeypatch.setattr(plan_mod, "SUPPORTED_BACKENDS", ("vllm", "sglang", "trtllm", "stubcfg2"))

    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="stub", address="127.0.0.1", port=49152,
    )
    _patch_externals(monkeypatch, tmp_path, vram_mb=20_000)

    configs_dir = tmp_path / "configs"
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"stubcfg2": _StubBackend()},
        models_dir=tmp_path, topology=topo_one_gpu,
        configs_dir=configs_dir,
    )
    model_store.add(conn, name="m", hf_repo="x/y")
    plan = replace(_make_plan(), model_name="m", hf_repo="x/y", backend="stubcfg2")
    dep = asyncio.run(mgr.load(plan))
    cfg = configs_dir / f"{dep.id}.yml"
    assert cfg.exists()

    asyncio.run(mgr.stop(dep.id))
    assert not cfg.exists(), "config file should be cleaned up on stop"
