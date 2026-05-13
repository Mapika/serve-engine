"""Daemon-lifespan task that runs the predictor on a fixed tick and
pre-warms adapters the rules picked.

v2.0 scope is intentionally narrow: only ADAPTER candidates trigger a
preload, and only when a ready deployment of the base already exists.
Pre-warming a base from scratch needs plan reconstruction (operator
chose gpu_ids / ctx / max_loras when they ran `serve run`); deferred.

Guardrails per design §5: never preempt an in-flight request, never
evict pinned, cap at max_prewarm_per_tick. Predictions are advisory —
a real request always wins.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

from serve_engine.backends.base import Backend
from serve_engine.lifecycle.adapter_router import (
    ensure_adapter_loaded,
    find_deployment_for,
)
from serve_engine.lifecycle.predictor import Predictor, PredictorConfig
from serve_engine.store import adapters as ad_store
from serve_engine.store import deployment_adapters as da_store

log = logging.getLogger(__name__)


class PredictorTask:
    """Long-running tick loop. Owns a Predictor + manager reference and
    a per-tick budget; otherwise stateless across ticks."""

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        backends: dict[str, Backend],
        models_dir: Path,
        config: PredictorConfig | None = None,
    ):
        self._conn = conn
        self._backends = backends
        self._models_dir = models_dir
        self._config = config or PredictorConfig()
        self._predictor = Predictor(conn, config=self._config)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # Stats surfaced via the daemon (`serve predict --history` will read
        # these once C-CLI lands; for now they're internal accounting).
        self.preloads_attempted = 0
        self.preloads_succeeded = 0
        self.preloads_skipped_already_warm = 0
        self.preloads_skipped_no_deployment = 0

    async def tick_once(self) -> int:
        """Run one prediction pass; return the number of preloads
        actually triggered. Caller wraps in exception handling."""
        if not self._config.enabled:
            return 0
        candidates = self._predictor.candidates()
        triggered = 0
        budget = self._config.max_prewarm_per_tick
        for c in candidates:
            if triggered >= budget:
                break
            if c.adapter_name is None:
                # Base pre-warming not in scope; bare-base candidates
                # are silently dropped from this tick. They'll still
                # serve traffic via the normal `serve run` path.
                continue
            dep = find_deployment_for(self._conn, c.base_name, c.adapter_name)
            if dep is None or dep.container_address is None:
                # No ready base deployment of this adapter's base, OR
                # the base has --max-loras=0. Adapter pre-warming is
                # impossible without an engine to load it into.
                self.preloads_skipped_no_deployment += 1
                continue
            a = ad_store.get_by_name(self._conn, c.adapter_name)
            if a is None:
                continue
            if dep.id in da_store.find_deployments_with_adapter(self._conn, a.id):
                self.preloads_skipped_already_warm += 1
                continue
            backend = self._backends.get(dep.backend)
            if backend is None:
                continue
            self.preloads_attempted += 1
            try:
                await ensure_adapter_loaded(
                    self._conn, backend, dep, c.adapter_name,
                    models_dir=self._models_dir,
                )
                self.preloads_succeeded += 1
                triggered += 1
                log.info(
                    "predictor preloaded adapter %r into dep #%d (%s)",
                    c.adapter_name, dep.id, c.reason,
                )
            except Exception as e:
                log.warning(
                    "predictor failed to preload %r into dep #%d: %s",
                    c.adapter_name, dep.id, e,
                )
        return triggered

    async def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.tick_once()
            except Exception:
                log.exception("predictor tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(self._config.tick_interval_s),
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
