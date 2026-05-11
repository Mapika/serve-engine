from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI

from serve_engine.backends.base import Backend
from serve_engine.daemon.admin import router as admin_router
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.lifecycle.manager import LifecycleManager


def build_app(
    *,
    conn: sqlite3.Connection,
    docker_client: DockerClient,
    backends: dict[str, Backend],
    models_dir: Path,
) -> FastAPI:
    app = FastAPI(title="serve-engine", version="0.0.1")
    app.state.conn = conn
    app.state.backends = backends
    app.state.manager = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends=backends,
        models_dir=models_dir,
    )
    app.include_router(admin_router)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app
