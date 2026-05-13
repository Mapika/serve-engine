from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from serve_engine.backends.vllm import VLLMBackend
from serve_engine.daemon.app import build_app
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.store import db


@pytest.fixture
def app(tmp_path, monkeypatch):
    from serve_engine.lifecycle.topology import GPUInfo, Topology

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
        lambda inp: 20_000,
    )
    (tmp_path / "weights").mkdir(exist_ok=True)

    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="127.0.0.1", port=49152,
    )

    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)

    topology = Topology(
        gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024)],
        _islands={0: frozenset({0})},
    )
    return build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
        topology=topology,
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
async def test_create_deployment_passes_extra_args_to_argv(app):
    # The fixture's docker_client is a MagicMock; we capture argv via its
    # .run side-effect so we can assert request-body extra_args reach the engine.
    docker_client = app.state.manager._docker  # injected MagicMock
    captured: dict[str, list[str]] = {}

    def _spy(**kwargs):
        captured["command"] = list(kwargs["command"])
        return ContainerHandle(id="cid", name="x", address="127.0.0.1", port=49152)

    docker_client.run.side_effect = _spy

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post(
            "/admin/deployments",
            json={
                "model_name": "qwen36",
                "hf_repo": "Qwen/Qwen3.6-35B-A3B-FP8",
                "image_tag": "vllm/vllm-openai:v0.20.2",
                "gpu_ids": [0],
                "max_model_len": 65536,
                "extra_args": {
                    "--kv-cache-dtype": "fp8_e4m3",
                    "--reasoning-parser": "qwen3",
                    "--enable-expert-parallel": "",
                },
            },
        )
    assert r.status_code == 201, r.text
    argv = captured["command"]
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    bare_idx = argv.index("--enable-expert-parallel")
    if bare_idx + 1 < len(argv):
        assert argv[bare_idx + 1].startswith("--")
    assert "" not in argv


@pytest.mark.asyncio
async def test_predictor_candidates_endpoint(app):
    """/admin/predictor/candidates returns a list of {model, score, reason}.
    With a fresh empty DB the list is empty — the endpoint should not 500."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/admin/predictor/candidates")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_predictor_stats_endpoint(app):
    """/admin/predictor/stats returns the tick-loop counters even before
    the first tick has run."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/admin/predictor/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["preloads_attempted"] == 0
    assert body["preloads_succeeded"] == 0
    assert "enabled" in body


@pytest.mark.asyncio
async def test_snapshot_endpoints_list_delete_gc(app, tmp_path):
    """End-to-end /admin/snapshots: list returns rows, DELETE by key wipes
    the row + on-disk dir, gc honors keep_last_per_model and removes the
    blobs of evicted snapshots."""
    from serve_engine.store import snapshots as snap_store

    conn = app.state.conn

    # Seed three snapshots for the same engine/model so GC keep_last=1
    # removes the two older ones.
    snap_root = tmp_path / "snapshots"
    snap_root.mkdir()
    ids: list[int] = []
    for i in range(3):
        d = snap_root / f"key{i}"
        d.mkdir()
        (d / "torch_cache").mkdir()
        (d / "torch_cache" / "blob.bin").write_bytes(b"x" * 1024)
        row = snap_store.add(
            conn, key=f"key{i}",
            hf_repo="org/model", revision="main",
            engine="vllm", engine_image="vllm/vllm-openai:v0.20.2",
            gpu_arch="9.0", quantization=None,
            max_model_len=4096, dtype="auto",
            tensor_parallel=1, target_concurrency=8,
            local_path=str(d), size_mb=1,
        )
        ids.append(row.id)
        # Spread last_used_at so list ordering is deterministic.
        conn.execute(
            "UPDATE snapshots SET last_used_at = datetime('now', ?) WHERE id=?",
            (f"-{i * 10} minutes", row.id),
        )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=10) as c:
        r = await c.get("/admin/snapshots")
        assert r.status_code == 200
        listed = r.json()
        assert len(listed) == 3
        # First entry should be the newest (key0 — last_used_at most recent).
        assert listed[0]["key"] == "key0"
        assert listed[0]["key_prefix"] == "key0"

        # Delete key2 explicitly; its directory must be gone afterward.
        d2 = snap_root / "key2"
        r = await c.delete("/admin/snapshots/key2")
        assert r.status_code == 204, r.text
        assert not d2.exists()
        assert snap_store.get_by_key(conn, "key2") is None

        # GC with keep_last=1 should drop key1 (we already deleted key2).
        r = await c.post(
            "/admin/snapshots/gc",
            json={"keep_last_per_model": 1},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["removed"] == 1
        assert snap_store.get_by_key(conn, "key1") is None
        assert snap_store.get_by_key(conn, "key0") is not None
        # Disk dir for key1 should be gone too.
        assert not (snap_root / "key1").exists()


@pytest.mark.asyncio
async def test_create_deployment_409_when_replacing_pinned(app):
    """If a same-name deployment is already pinned, the daemon must return
    a 4xx with the manager's "is pinned" message — not a 500 — so the CLI
    can show the actionable hint ("run `serve unpin <model>`")."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r1 = await c.post(
            "/admin/deployments",
            json={
                "model_name": "llama-1b",
                "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
                "image_tag": "vllm/vllm-openai:v0.7.3",
                "gpu_ids": [0], "max_model_len": 8192, "pinned": True,
            },
        )
        assert r1.status_code == 201, r1.text
        r2 = await c.post(
            "/admin/deployments",
            json={
                "model_name": "llama-1b",
                "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
                "image_tag": "vllm/vllm-openai:v0.7.3",
                "gpu_ids": [0], "max_model_len": 8192,
            },
        )
    assert r2.status_code == 409, r2.text
    body = r2.json()
    assert "is pinned" in body["detail"]
    assert "serve unpin" in body["detail"]


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


@pytest.mark.asyncio
async def test_pin_unpin_deployment(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30,
    ) as c:
        r = await c.post(
            "/admin/deployments",
            json={
                "model_name": "x",
                "hf_repo": "org/x",
                "image_tag": "img:v1",
                "gpu_ids": [0],
                "max_model_len": 4096,
            },
        )
        dep_id = r.json()["id"]

        r = await c.post(f"/admin/deployments/{dep_id}/pin")
        assert r.status_code == 204

        r = await c.get("/admin/deployments")
        assert r.json()[0]["pinned"] is True

        r = await c.post(f"/admin/deployments/{dep_id}/unpin")
        assert r.status_code == 204
        r = await c.get("/admin/deployments")
        assert r.json()[0]["pinned"] is False


@pytest.mark.asyncio
async def test_pin_404(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test",
    ) as c:
        r = await c.post("/admin/deployments/999/pin")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_deployment_by_id(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30,
    ) as c:
        r = await c.post(
            "/admin/deployments",
            json={
                "model_name": "x",
                "hf_repo": "org/x",
                "image_tag": "img:v1",
                "gpu_ids": [0],
                "max_model_len": 4096,
            },
        )
        dep_id = r.json()["id"]
        r = await c.delete(f"/admin/deployments/{dep_id}")
        assert r.status_code == 204
        r = await c.get("/admin/deployments")
        deps = r.json()
        # Deployment row still exists but in stopped status
        assert deps[0]["status"] == "stopped"


@pytest.mark.asyncio
async def test_create_deployment_default_backend_is_vllm(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post(
            "/admin/deployments",
            json={
                "model_name": "x",
                "hf_repo": "org/x",
                "image_tag": "img:v1",
                "gpu_ids": [0],
                "max_model_len": 4096,
                # no `backend` field — should default via selection
            },
        )
    assert r.status_code == 201
    body = r.json()
    assert body["backend"] == "vllm"


@pytest.mark.asyncio
async def test_list_gpus_returns_list(app, monkeypatch):
    from serve_engine.observability.gpu_stats import GPUSnapshot
    monkeypatch.setattr(
        "serve_engine.daemon.admin._read_gpu_stats",
        lambda: [GPUSnapshot(
            index=0, memory_used_mb=10_000, memory_total_mb=80_000,
            gpu_util_pct=42, power_w=350,
        )],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/admin/gpus")
    assert r.status_code == 200
    rows = r.json()
    assert rows[0]["index"] == 0
    assert rows[0]["gpu_util_pct"] == 42


@pytest.mark.asyncio
async def test_create_list_revoke_key(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        # Create admin key first (bypass is active at this point — no keys exist yet)
        r = await c.post("/admin/keys", json={"name": "alice", "tier": "admin"})
        assert r.status_code == 201
        body = r.json()
        assert body["secret"].startswith("sk-")
        kid = body["id"]
        secret = body["secret"]
        auth = {"Authorization": f"Bearer {secret}"}

        # Now a key exists — all subsequent admin requests must carry the bearer
        r = await c.get("/admin/keys", headers=auth)
        assert r.status_code == 200
        names = [k["name"] for k in r.json()]
        assert "alice" in names

        r = await c.delete(f"/admin/keys/{kid}", headers=auth)
        assert r.status_code == 204

        r = await c.get("/admin/keys", headers=auth)
        revoked = [k for k in r.json() if k["id"] == kid]
        assert revoked[0]["revoked"] is True


@pytest.mark.asyncio
async def test_download_model_endpoint(app, monkeypatch, tmp_path):
    """POST /admin/models/{name}/download invokes the downloader."""
    captured = {}

    def fake_download_model(*, hf_repo, revision, cache_dir):
        captured["hf_repo"] = hf_repo
        captured["revision"] = revision
        path = tmp_path / "fake_weights"
        path.mkdir(exist_ok=True)
        return str(path)

    monkeypatch.setattr(
        "serve_engine.lifecycle.downloader.download_model",
        fake_download_model,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30,
    ) as c:
        # Register first
        r = await c.post("/admin/models", json={"name": "x", "hf_repo": "org/x"})
        assert r.status_code == 201
        # Trigger download
        r = await c.post("/admin/models/x/download")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "x"
        assert body["local_path"].endswith("fake_weights")
        assert body["already_present"] is False

        # Second call returns already_present=True
        r = await c.post("/admin/models/x/download")
        assert r.json()["already_present"] is True


@pytest.mark.asyncio
async def test_download_unknown_model_404(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/admin/models/no-such/download")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_admin_route_requires_admin_tier(tmp_path, monkeypatch):
    """When non-admin keys exist, admin routes return 403 unless the bearer is admin."""
    from serve_engine.backends.vllm import VLLMBackend
    from serve_engine.daemon.app import build_app
    from serve_engine.lifecycle.docker_client import ContainerHandle
    from serve_engine.lifecycle.topology import GPUInfo, Topology
    from serve_engine.store import api_keys as _ak
    from serve_engine.store import db as _db

    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.estimate_vram_mb",
        lambda inp: 20_000,
    )
    (tmp_path / "weights").mkdir(exist_ok=True)
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="127.0.0.1", port=49152,
    )
    conn = _db.connect(tmp_path / "t.db")
    _db.init_schema(conn)
    # Create a standard-tier key so the bypass is off
    std_secret, _ = _ak.create(conn, name="user", tier="standard")
    admin_secret, _ = _ak.create(conn, name="root", tier="admin")
    topology = Topology(
        gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024)],
        _islands={0: frozenset({0})},
    )
    app = build_app(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path, topology=topology,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        # No bearer → 401
        r = await c.get("/admin/models")
        assert r.status_code == 401

        # Standard-tier bearer → 403
        r = await c.get(
            "/admin/models",
            headers={"Authorization": f"Bearer {std_secret}"},
        )
        assert r.status_code == 403

        # Admin-tier bearer → 200
        r = await c.get(
            "/admin/models",
            headers={"Authorization": f"Bearer {admin_secret}"},
        )
        assert r.status_code == 200
