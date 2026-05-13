from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from serve_engine.auth.tiers import load_tiers
from serve_engine.backends.base import Backend
from serve_engine.daemon.admin import router as admin_router
from serve_engine.daemon.metrics_router import router as metrics_router
from serve_engine.daemon.openai_proxy import router as openai_router
from serve_engine.daemon.ui_router import install_ui
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.topology import Topology
from serve_engine.observability.events import EventBus

log = logging.getLogger(__name__)


def _attach_state(
    app: FastAPI,
    *,
    conn: sqlite3.Connection,
    backends: dict[str, Backend],
    manager: LifecycleManager,
    event_bus: EventBus,
) -> None:
    app.state.conn = conn
    app.state.backends = backends
    app.state.manager = manager
    app.state.event_bus = event_bus
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
    configs_dir: Path | None = None,
    snapshots_dir: Path | None = None,
) -> tuple[FastAPI, FastAPI]:
    """Returns (tcp_app, uds_app) sharing the same LifecycleManager.

    - tcp_app: public OpenAI-compatible API + admin routes. Owns the lifespan
      (reconcile on startup, stop_all on shutdown).
    - uds_app: full surface (admin + OpenAI) for the local CLI / future UI.
      No separate lifespan — shares the single Reaper and manager with tcp_app.
    """
    event_bus = EventBus()
    manager = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends=backends,
        models_dir=models_dir,
        topology=topology,
        event_bus=event_bus,
        configs_dir=configs_dir,
        snapshots_dir=snapshots_dir,
    )

    from serve_engine.lifecycle.reaper import Reaper
    from serve_engine.lifecycle.snapshot_gc import SnapshotGc
    from serve_engine.store import deployments as _dep_store
    reaper = Reaper(
        manager=manager,
        list_ready=lambda: _dep_store.list_ready(conn),
    )
    # Snapshot eviction loop. Reads ~/.serve/snapshots.yaml each tick so
    # operators can adjust keep_last_per_model / max_disk_gb without a
    # daemon restart. Defaults to keep_last_per_model=2 + no disk cap;
    # ticks every 6 hours.
    from serve_engine import config as _cfg
    snapshot_gc = SnapshotGc(
        conn=conn,
        cfg_path=_cfg.SERVE_DIR / "snapshots.yaml",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Startup
        try:
            await manager.reconcile()
        except Exception:
            log.exception("reconcile failed; continuing")
        # Run one GC pass at startup so a restarted daemon catches up on
        # any snapshots that landed but exceeded policy since last tick.
        try:
            result = await snapshot_gc.tick_once()
            if result["removed"] > 0:
                log.info(
                    "startup snapshot gc: removed %d, %d MB remaining",
                    result["removed"], result["remaining_mb"],
                )
        except Exception:
            log.exception("startup snapshot gc failed; continuing")
        reaper.start()
        snapshot_gc.start()
        yield
        # Shutdown
        await snapshot_gc.stop()
        await reaper.stop()
        try:
            await manager.stop_all()
        except Exception:
            log.exception("stop_all on shutdown failed")

    tcp_app = FastAPI(title="serve-engine (public)", version="0.0.1", lifespan=lifespan)
    _attach_state(tcp_app, conn=conn, backends=backends, manager=manager, event_bus=event_bus)
    tcp_app.include_router(openai_router)
    tcp_app.include_router(metrics_router)
    tcp_app.include_router(admin_router)
    install_ui(tcp_app)

    uds_app = FastAPI(title="serve-engine (control)", version="0.0.1")
    _attach_state(uds_app, conn=conn, backends=backends, manager=manager, event_bus=event_bus)
    uds_app.include_router(openai_router)
    uds_app.include_router(admin_router)
    uds_app.include_router(metrics_router)

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
