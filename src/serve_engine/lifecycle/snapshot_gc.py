"""Snapshot garbage collection — eviction logic + background loop.

Two policies stacked: keep at most N most-recently-used per (engine,
hf_repo), then a global LRU sweep if total disk still exceeds a cap.
The admin endpoint and the daemon-lifespan loop both call `run_gc` so
operators see identical behavior whether they triggered it manually or
the timer did.

Config lives in `~/.serve/snapshots.yaml` (defaults applied when
missing). See design §9.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from serve_engine.store import snapshots as snapshot_store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotGcConfig:
    keep_last_per_model: int = 2
    max_disk_gb: float | None = None
    tick_s: float = 6 * 3600.0  # 6 hours

    @classmethod
    def load(cls, path: Path) -> SnapshotGcConfig:
        """Read `~/.serve/snapshots.yaml`. Absent file → defaults."""
        if not path.is_file():
            return cls()
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            log.warning("failed to read %s: %s; using defaults", path, e)
            return cls()
        return cls(
            keep_last_per_model=int(data.get("keep_last_per_model", 2)),
            max_disk_gb=(
                float(data["max_disk_gb"])
                if data.get("max_disk_gb") is not None
                else None
            ),
            tick_s=float(data.get("tick_s", 6 * 3600.0)),
        )


def run_gc(
    conn: sqlite3.Connection,
    *,
    keep_last_per_model: int,
    max_disk_gb: float | None,
) -> dict:
    """Apply the two-step policy and return {removed, remaining_mb}.

    Step 1: for each (engine, hf_repo) pair, drop everything beyond the
    `keep_last_per_model` most-recently-used.
    Step 2: if `max_disk_gb` is set and the total still exceeds it,
    LRU-evict globally until under the cap.

    The DB row is the source of truth; the on-disk dir is removed best-
    effort (operator can clean leftover dirs manually if rmtree fails).
    """
    removed = 0
    if keep_last_per_model > 0:
        seen: set[tuple[str, str]] = set()
        for s in snapshot_store.list_all(conn):
            pair = (s.engine, s.hf_repo)
            if pair in seen:
                continue
            seen.add(pair)
            victims = snapshot_store.lru_for_engine_model(
                conn, s.engine, s.hf_repo, keep_n=keep_last_per_model,
            )
            for v in victims:
                _remove_blob(Path(v.local_path))
                snapshot_store.delete(conn, v.id)
                removed += 1
    if max_disk_gb is not None and max_disk_gb > 0:
        cap_mb = int(max_disk_gb * 1024)
        rows = list(snapshot_store.list_all(conn))
        while rows and snapshot_store.total_size_mb(conn) > cap_mb:
            victim = rows.pop()  # list_all is last_used_at DESC; pop = oldest
            _remove_blob(Path(victim.local_path))
            snapshot_store.delete(conn, victim.id)
            removed += 1
    return {
        "removed": removed,
        "remaining_mb": snapshot_store.total_size_mb(conn),
    }


def _remove_blob(path: Path) -> None:
    if path.exists():
        try:
            shutil.rmtree(path)
        except OSError:
            pass


class SnapshotGc:
    """Background snapshot eviction loop.

    Ticks every cfg.tick_s seconds (default 6h). Also exposes `tick_once`
    for the daemon's startup-time eviction and tests. Designed to share
    the lifespan pattern with Reaper.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        cfg_path: Path,
    ):
        self._conn = conn
        self._cfg_path = cfg_path
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def config(self) -> SnapshotGcConfig:
        # Re-read each tick so operators can edit snapshots.yaml without
        # restarting the daemon.
        return SnapshotGcConfig.load(self._cfg_path)

    async def tick_once(self) -> dict:
        cfg = self.config
        return run_gc(
            self._conn,
            keep_last_per_model=cfg.keep_last_per_model,
            max_disk_gb=cfg.max_disk_gb,
        )

    async def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = await self.tick_once()
                if result["removed"] > 0:
                    log.info(
                        "snapshot gc: removed %d snapshot(s); %d MB remaining",
                        result["removed"], result["remaining_mb"],
                    )
            except Exception:
                log.exception("snapshot gc tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.config.tick_s,
                )
            except TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
