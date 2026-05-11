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
