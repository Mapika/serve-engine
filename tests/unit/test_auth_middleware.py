import httpx
import pytest
from fastapi import Depends, FastAPI

from serve_engine.auth import tiers
from serve_engine.auth.middleware import require_auth_dep
from serve_engine.store import api_keys, db


@pytest.fixture
def app_factory(tmp_path):
    def make(create_admin_key: bool):
        conn = db.connect(tmp_path / "t.db")
        db.init_schema(conn)
        secret = None
        if create_admin_key:
            secret, _ = api_keys.create(conn, name="root", tier="admin")
        a = FastAPI()
        a.state.conn = conn
        a.state.tier_cfg = tiers.load_tiers()

        @a.post("/v1/test")
        async def _test(_=Depends(require_auth_dep)):
            return {"ok": True}

        return a, secret
    return make


@pytest.mark.asyncio
async def test_no_keys_table_empty_means_bypass(app_factory):
    """When the api_keys table is empty, auth is bypassed entirely."""
    app, _ = app_factory(create_admin_key=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/v1/test")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_missing_bearer_when_keys_exist_401(app_factory):
    app, _ = app_factory(create_admin_key=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/v1/test")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_bad_bearer_401(app_factory):
    app, _ = app_factory(create_admin_key=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/v1/test", headers={"Authorization": "Bearer sk-bogus"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_good_bearer_passes(app_factory):
    app, secret = app_factory(create_admin_key=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/v1/test", headers={"Authorization": f"Bearer {secret}"})
    assert r.status_code == 200
