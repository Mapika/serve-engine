from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI

from serve_engine.auth.tiers import load_tiers
from serve_engine.backends.base import Backend
from serve_engine.daemon.admin import router as admin_router
from serve_engine.daemon.metrics_router import router as metrics_router
from serve_engine.daemon.openai_proxy import router as openai_router
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.topology import Topology


def _attach_state(
    app: FastAPI,
    *,
    conn: sqlite3.Connection,
    backends: dict[str, Backend],
    manager: LifecycleManager,
) -> None:
    app.state.conn = conn
    app.state.backends = backends
    app.state.manager = manager
    app.state.tier_cfg = load_tiers()
    app.state.request_count = 0

    @app.get("/healthz")
    def healthz():
        return {"ok": True}


def build_apps(
    *,
    conn: sqlite3.Connection,
    docker_client: DockerClient,
    backends: dict[str, Backend],
    models_dir: Path,
    topology: Topology | None = None,
) -> tuple[FastAPI, FastAPI]:
    """Returns (tcp_app, uds_app) sharing the same LifecycleManager.

    - tcp_app: public OpenAI-compatible API only. No admin routes.
    - uds_app: full surface (admin + OpenAI) for the local CLI / future UI.
    """
    manager = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends=backends,
        models_dir=models_dir,
        topology=topology,
    )

    tcp_app = FastAPI(title="serve-engine (public)", version="0.0.1")
    _attach_state(tcp_app, conn=conn, backends=backends, manager=manager)
    tcp_app.include_router(openai_router)
    tcp_app.include_router(metrics_router)

    from serve_engine.lifecycle.reaper import Reaper
    from serve_engine.store import deployments as _dep_store
    reaper = Reaper(
        manager=manager,
        list_ready=lambda: _dep_store.list_ready(conn),
    )

    uds_app = FastAPI(title="serve-engine (control)", version="0.0.1")
    _attach_state(uds_app, conn=conn, backends=backends, manager=manager)
    uds_app.include_router(openai_router)
    uds_app.include_router(admin_router)
    uds_app.include_router(metrics_router)

    @uds_app.on_event("startup")
    async def _start_reaper() -> None:
        reaper.start()

    @uds_app.on_event("shutdown")
    async def _stop_reaper() -> None:
        await reaper.stop()

    return tcp_app, uds_app


def build_app(
    *,
    conn: sqlite3.Connection,
    docker_client: DockerClient,
    backends: dict[str, Backend],
    models_dir: Path,
    topology: Topology | None = None,
) -> FastAPI:
    """Single-app factory retained for tests that exercise the full surface."""
    _, uds_app = build_apps(
        conn=conn,
        docker_client=docker_client,
        backends=backends,
        models_dir=models_dir,
        topology=topology,
    )
    return uds_app
