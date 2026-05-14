# Serving Engine - Plan 05: Observability

**Goal:** Make the daemon legible. Three surfaces: `GET /metrics` (Prometheus-format, aggregated from engines + daemon), `GET /admin/events` (SSE event stream for lifecycle transitions), `GET /admin/gpus` (live per-GPU memory and utilization). One new CLI command: `serve top` - htop-style live view of deployments, GPUs, and request rate. No new third-party libraries; we generate Prometheus text by hand and use the existing `pynvml` for GPU stats.

**Architecture:** A small in-process event bus (`asyncio.Queue` fanout) collects lifecycle events emitted by `LifecycleManager`. The SSE endpoint subscribes per-request and forwards. `/metrics` proxies each engine's `/metrics` and concatenates them with daemon-level metrics (requests served, deployments by state, etc.). `serve top` uses the events stream + a poll on `/admin/gpus` and `/admin/deployments`.

**Tech Stack:** Same as Plans 01-04. New: `rich` for the terminal UI in `serve top` (already pulled in transitively by `typer`/`structlog`, but added explicitly).

---

## File structure

```
serving-engine/
|-- src/serve_engine/
|   |-- observability/
|   |   |-- __init__.py
|   |   |-- events.py          # in-process pub/sub
|   |   |-- metrics.py         # Prometheus text generation + engine aggregation
|   |   +-- gpu_stats.py       # pynvml-backed per-GPU snapshot
|   |-- lifecycle/
|   |   +-- manager.py         # MODIFIED - emit events on transitions
|   |-- daemon/
|   |   |-- admin.py           # MODIFIED - /admin/gpus + /admin/events
|   |   |-- app.py             # MODIFIED - instantiate event bus, expose /metrics
|   |   +-- metrics_router.py  # NEW - /metrics route
|   +-- cli/
|       +-- top_cmd.py         # NEW - serve top
+-- tests/
    +-- unit/
        |-- test_events.py
        |-- test_metrics.py
        |-- test_gpu_stats.py
        +-- test_admin_events_endpoint.py
```

---

## Task 1: In-process event bus

**Files:** `src/serve_engine/observability/__init__.py` (empty), `src/serve_engine/observability/events.py`, `tests/unit/test_events.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_events.py`:
```python
import asyncio

import pytest

from serve_engine.observability.events import Event, EventBus


@pytest.mark.asyncio
async def test_subscribe_receives_published():
    bus = EventBus()
    received: list[Event] = []
    async with bus.subscribe() as queue:
        await bus.publish(Event(kind="load.started", payload={"dep_id": 1}))
        e = await asyncio.wait_for(queue.get(), timeout=1.0)
        received.append(e)
    assert received[0].kind == "load.started"
    assert received[0].payload == {"dep_id": 1}


@pytest.mark.asyncio
async def test_multiple_subscribers_each_receive():
    bus = EventBus()
    async with bus.subscribe() as q1, bus.subscribe() as q2:
        await bus.publish(Event(kind="x", payload={}))
        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e1.kind == "x" and e2.kind == "x"


@pytest.mark.asyncio
async def test_subscriber_unsubscribes_on_exit():
    bus = EventBus()
    async with bus.subscribe() as _:
        assert bus.subscriber_count() == 1
    assert bus.subscriber_count() == 0
```

- [ ] **Step 2: Implement `events.py`**

```python
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EventBus:
    """Tiny asyncio fanout. Each subscriber gets its own queue; backpressure on
    a slow subscriber doesn't block publishing to others (we drop on full)."""

    def __init__(self, *, per_subscriber_buffer: int = 256):
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._buf = per_subscriber_buffer

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @contextlib.asynccontextmanager
    async def subscribe(self):
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._buf)
        self._subscribers.add(q)
        try:
            yield q
        finally:
            self._subscribers.discard(q)

    async def publish(self, event: Event) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop on slow consumer; logs/metrics will reflect the lag.
                pass
```

- [ ] **Step 3: Run + commit**

```bash
source .venv/bin/activate
pytest tests/unit/test_events.py -v
ruff check src/ tests/
git add src/serve_engine/observability/ tests/unit/test_events.py
git commit -m "feat(observability): in-process EventBus with async fanout"
```

---

## Task 2: GPU stats snapshot

**Files:** `src/serve_engine/observability/gpu_stats.py`, `tests/unit/test_gpu_stats.py`

- [ ] **Step 1: Tests with pynvml mocked**

```python
from unittest.mock import MagicMock, patch

from serve_engine.observability.gpu_stats import GPUSnapshot, read_gpu_stats


@patch("serve_engine.observability.gpu_stats.pynvml")
def test_read_gpu_stats(mock_nvml):
    mock_nvml.nvmlInit = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    handle = MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
        used=20 * 1024**3, total=80 * 1024**3,
    )
    util = MagicMock(); util.gpu = 42
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = util
    mock_nvml.nvmlDeviceGetPowerUsage.return_value = 350_000  # mW

    snaps = read_gpu_stats()
    assert len(snaps) == 1
    s = snaps[0]
    assert s.index == 0
    assert s.memory_used_mb == 20 * 1024
    assert s.memory_total_mb == 80 * 1024
    assert s.gpu_util_pct == 42
    assert s.power_w == 350


@patch("serve_engine.observability.gpu_stats.pynvml", None)
def test_read_gpu_stats_without_pynvml():
    assert read_gpu_stats() == []
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations

from dataclasses import dataclass

try:
    import pynvml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GPUSnapshot:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    gpu_util_pct: int
    power_w: int


def read_gpu_stats() -> list[GPUSnapshot]:
    """Live per-GPU memory + utilization + power. Empty list if pynvml absent."""
    if pynvml is None:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    out: list[GPUSnapshot] = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
        except Exception:
            power_mw = 0
        out.append(GPUSnapshot(
            index=i,
            memory_used_mb=int(mem.used) // 1024 // 1024,
            memory_total_mb=int(mem.total) // 1024 // 1024,
            gpu_util_pct=int(util.gpu),
            power_w=int(power_mw) // 1000,
        ))
    return out
```

- [ ] **Step 3: Commit**

```bash
pytest tests/unit/test_gpu_stats.py -v
ruff check src/ tests/
git add src/serve_engine/observability/gpu_stats.py tests/unit/test_gpu_stats.py
git commit -m "feat(observability): per-GPU live snapshot via pynvml"
```

---

## Task 3: Prometheus metrics generation

**Files:** `src/serve_engine/observability/metrics.py`, `src/serve_engine/daemon/metrics_router.py`, `tests/unit/test_metrics.py`

The daemon's `/metrics` returns a single Prometheus text body that concatenates:
1. Daemon-level metrics (deployment counts by state, registered model count, active key count, request counters).
2. Per-deployment metrics fetched from each engine's `/metrics` endpoint, with engine container address replaced by our deployment id.

- [ ] **Step 1: Tests for `format_daemon_metrics`**

```python
from serve_engine.observability.metrics import format_daemon_metrics


def test_format_daemon_metrics_empty():
    text = format_daemon_metrics(
        deployments_by_status={},
        models_total=0,
        api_keys_active=0,
        request_count=0,
    )
    assert "# TYPE serve_deployments gauge" in text
    assert "serve_deployments{status=" not in text  # no rows
    assert "serve_models_total 0" in text
    assert "serve_api_keys_active 0" in text
    assert "serve_proxy_requests_total 0" in text


def test_format_daemon_metrics_with_rows():
    text = format_daemon_metrics(
        deployments_by_status={"ready": 2, "loading": 1},
        models_total=3,
        api_keys_active=5,
        request_count=42,
    )
    assert 'serve_deployments{status="ready"} 2' in text
    assert 'serve_deployments{status="loading"} 1' in text
    assert "serve_models_total 3" in text
    assert "serve_api_keys_active 5" in text
    assert "serve_proxy_requests_total 42" in text
```

- [ ] **Step 2: Implement `src/serve_engine/observability/metrics.py`**

```python
from __future__ import annotations

import asyncio

import httpx


def format_daemon_metrics(
    *,
    deployments_by_status: dict[str, int],
    models_total: int,
    api_keys_active: int,
    request_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# HELP serve_deployments Count of deployments by status.")
    lines.append("# TYPE serve_deployments gauge")
    for status, n in sorted(deployments_by_status.items()):
        lines.append(f'serve_deployments{{status="{status}"}} {n}')
    lines.append("# HELP serve_models_total Number of registered models.")
    lines.append("# TYPE serve_models_total gauge")
    lines.append(f"serve_models_total {models_total}")
    lines.append("# HELP serve_api_keys_active Number of non-revoked API keys.")
    lines.append("# TYPE serve_api_keys_active gauge")
    lines.append(f"serve_api_keys_active {api_keys_active}")
    lines.append("# HELP serve_proxy_requests_total Total /v1/* requests processed.")
    lines.append("# TYPE serve_proxy_requests_total counter")
    lines.append(f"serve_proxy_requests_total {request_count}")
    return "\n".join(lines) + "\n"


async def fetch_engine_metrics(base_url: str, path: str = "/metrics") -> str:
    """Best-effort fetch of an engine's Prometheus metrics. Returns '' on failure."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(base_url.rstrip("/") + path)
            if r.status_code == 200:
                return r.text
    except httpx.HTTPError:
        pass
    return ""


async def gather_engine_metrics(engine_urls: list[tuple[int, str]]) -> str:
    """engine_urls is [(deployment_id, base_url)]. Concatenates with a header per dep."""
    if not engine_urls:
        return ""
    bodies = await asyncio.gather(
        *(fetch_engine_metrics(url) for _, url in engine_urls),
        return_exceptions=False,
    )
    out: list[str] = []
    for (dep_id, _), body in zip(engine_urls, bodies):
        if not body:
            continue
        out.append(f"# --- deployment {dep_id} ---")
        out.append(body.rstrip())
    return "\n".join(out) + ("\n" if out else "")
```

- [ ] **Step 3: Implement `src/serve_engine/daemon/metrics_router.py`**

```python
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from serve_engine.observability.metrics import (
    fetch_engine_metrics,
    format_daemon_metrics,
    gather_engine_metrics,
)
from serve_engine.store import api_keys as _ak_store
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store


router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request) -> str:
    conn: sqlite3.Connection = request.app.state.conn
    by_status: dict[str, int] = {}
    for d in dep_store.list_all(conn):
        by_status[d.status] = by_status.get(d.status, 0) + 1
    daemon_text = format_daemon_metrics(
        deployments_by_status=by_status,
        models_total=len(model_store.list_all(conn)),
        api_keys_active=_ak_store.count_active(conn),
        request_count=getattr(request.app.state, "request_count", 0),
    )

    # Engine metrics
    backends_dict = request.app.state.backends
    engine_urls: list[tuple[int, str]] = []
    for d in dep_store.list_ready(conn):
        backend = backends_dict.get(d.backend)
        if backend is None or d.container_address is None:
            continue
        url = f"http://{d.container_address}:{d.container_port}{backend.metrics_path}"
        engine_urls.append((d.id, url))
    engine_text = await gather_engine_metrics(engine_urls)
    return daemon_text + engine_text
```

- [ ] **Step 4: Mount the router in `src/serve_engine/daemon/app.py`**

Add to imports:
```python
from serve_engine.daemon.metrics_router import router as metrics_router
```

In `build_apps`, on BOTH `tcp_app` and `uds_app`, after the OpenAI/admin routers, add:
```python
    tcp_app.include_router(metrics_router)
    uds_app.include_router(metrics_router)
```

Also initialize the counter in `_attach_state`:
```python
    app.state.request_count = 0
```

- [ ] **Step 5: Wire the counter - increment in proxy**

In `daemon/openai_proxy.py` `_proxy`, after `dep_store.touch_last_request(conn, active.id)`, add:
```python
    request.app.state.request_count = getattr(request.app.state, "request_count", 0) + 1
```

- [ ] **Step 6: Run + commit**

```bash
pytest tests/unit/test_metrics.py -v
pytest -v   # full suite
ruff check src/ tests/
git add src/serve_engine/observability/metrics.py src/serve_engine/daemon/metrics_router.py src/serve_engine/daemon/app.py src/serve_engine/daemon/openai_proxy.py tests/unit/test_metrics.py
git commit -m "feat(observability): /metrics endpoint aggregating daemon + engine Prometheus"
```

---

## Task 4: Lifecycle event emission + /admin/events SSE

**Files:** `src/serve_engine/lifecycle/manager.py` (modify), `src/serve_engine/daemon/admin.py` (modify), `src/serve_engine/daemon/app.py` (modify), `tests/unit/test_admin_events_endpoint.py`

- [ ] **Step 1: Modify `LifecycleManager` to accept an EventBus and emit on each transition**

In `src/serve_engine/lifecycle/manager.py`:

Add import:
```python
from serve_engine.observability.events import Event, EventBus
```

In `__init__`, add a new keyword parameter `event_bus: EventBus | None = None` (default None for backward compat / tests). Store as `self._events = event_bus`.

Add a helper inside the class:
```python
    async def _emit(self, kind: str, **payload) -> None:
        if self._events is not None:
            await self._events.publish(Event(kind=kind, payload=payload))
```

Emit events at these points in `load()`:
- Just after `dep_store.update_status(...,"loading")`: `await self._emit("deployment.loading", dep_id=dep.id, model=plan.model_name, backend=plan.backend)`
- Right before `wait_healthy`: `await self._emit("deployment.spawned", dep_id=dep.id, container_id=handle.id)`
- After `update_status(...,"ready")`: `await self._emit("deployment.ready", dep_id=dep.id)`
- In the failed branch right before raising: `await self._emit("deployment.failed", dep_id=dep.id, error=msg)`

In `_stop_locked`:
- After `update_status(...,"stopped")`: `await self._emit("deployment.stopped", dep_id=dep_id)`

In `pin`:
- After updating: `await self._emit("deployment.pinned" if pinned else "deployment.unpinned", dep_id=dep_id)`

- [ ] **Step 2: Wire `EventBus` into `build_apps`**

In `src/serve_engine/daemon/app.py`, add to imports:
```python
from serve_engine.observability.events import EventBus
```

In `build_apps`, before constructing `LifecycleManager`:
```python
    event_bus = EventBus()
```

Pass `event_bus=event_bus` to `LifecycleManager(...)`. Then in `_attach_state`, add:
```python
    app.state.event_bus = event_bus
```

(Both `tcp_app` and `uds_app` get the same bus - but only `uds_app` will expose `/admin/events`.)

- [ ] **Step 3: SSE endpoint in `admin.py`**

Append to `src/serve_engine/daemon/admin.py`:
```python
import json as _json

from fastapi.responses import StreamingResponse as _SSE


@router.get("/events")
async def events(request: Request) -> _SSE:
    """SSE: lifecycle events as `data: <json>\n\n` chunks. Heartbeat every 15s."""
    bus = request.app.state.event_bus

    async def gen():
        async with bus.subscribe() as queue:
            yield ":ok\n\n"  # initial heartbeat
            while True:
                try:
                    e = await asyncio.wait_for(queue.get(), timeout=15.0)
                    payload = _json.dumps({
                        "kind": e.kind, "payload": e.payload, "ts": e.ts,
                    })
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ":hb\n\n"  # SSE comment heartbeat

    return _SSE(gen(), media_type="text/event-stream")
```

Also add `import asyncio` at the top of `admin.py` if not already imported.

- [ ] **Step 4: Test**

`tests/unit/test_admin_events_endpoint.py`:
```python
import asyncio

import httpx
import pytest

from serve_engine.daemon.app import build_app
from serve_engine.observability.events import Event
from serve_engine.store import db


@pytest.fixture
def app_with_bus(tmp_path):
    from serve_engine.backends.vllm import VLLMBackend
    from serve_engine.lifecycle.docker_client import ContainerHandle
    from serve_engine.lifecycle.topology import GPUInfo, Topology
    from unittest.mock import MagicMock

    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="127.0.0.1", port=49152
    )
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
async def test_events_endpoint_streams_published_events(app_with_bus):
    bus = app_with_bus.state.event_bus
    transport = httpx.ASGITransport(app=app_with_bus)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5) as c:
        async def consume():
            async with c.stream("GET", "/admin/events") as r:
                lines: list[str] = []
                async for line in r.aiter_lines():
                    lines.append(line)
                    if len(lines) >= 4:
                        break
                return lines
        # Publish after a beat so the subscriber is ready
        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await bus.publish(Event(kind="test.fired", payload={"x": 1}))
        lines = await asyncio.wait_for(consumer, timeout=2.0)
    # Expect the initial ":ok" comment then a data: line containing test.fired
    assert any("test.fired" in line for line in lines)
```

- [ ] **Step 5: Run + commit**

```bash
pytest tests/unit/test_admin_events_endpoint.py -v
pytest -v
ruff check src/ tests/
git add src/serve_engine/lifecycle/manager.py src/serve_engine/daemon/admin.py src/serve_engine/daemon/app.py tests/unit/test_admin_events_endpoint.py
git commit -m "feat(observability): /admin/events SSE + manager event emission"
```

---

## Task 5: `/admin/gpus` endpoint

**Files:** `src/serve_engine/daemon/admin.py` (modify)

- [ ] **Step 1: Add the route**

Append to `admin.py`:
```python
from serve_engine.observability.gpu_stats import read_gpu_stats as _read_gpu_stats


@router.get("/gpus")
def list_gpus():
    """Per-GPU live snapshot: memory, utilization, power."""
    return [
        {
            "index": s.index,
            "memory_used_mb": s.memory_used_mb,
            "memory_total_mb": s.memory_total_mb,
            "gpu_util_pct": s.gpu_util_pct,
            "power_w": s.power_w,
        }
        for s in _read_gpu_stats()
    ]
```

- [ ] **Step 2: Quick test**

Append to `tests/unit/test_admin_endpoints.py`:

```python
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
```

- [ ] **Step 3: Run + commit**

```bash
pytest -v
ruff check src/ tests/
git add src/serve_engine/daemon/admin.py tests/unit/test_admin_endpoints.py
git commit -m "feat(observability): /admin/gpus per-GPU snapshot endpoint"
```

---

## Task 6: `serve top` CLI

**Files:** `src/serve_engine/cli/top_cmd.py`, `src/serve_engine/cli/__init__.py` (register)

- [ ] **Step 1: Implement `top_cmd.py`**

```python
from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from serve_engine import config
from serve_engine.cli import app


@app.command("top")
def top(refresh_s: float = typer.Option(1.0, "--refresh", "-r")):
    """Live view of deployments, GPUs, and recent events."""
    console = Console()
    asyncio.run(_run(console, refresh_s))


async def _run(console: Console, refresh_s: float) -> None:
    transport = httpx.AsyncHTTPTransport(uds=str(config.SOCK_PATH))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://daemon", timeout=None,
    ) as c:
        last_events: list[dict] = []

        async def consume_events():
            try:
                async with c.stream("GET", "/admin/events") as r:
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                obj = json.loads(line[len("data:"):].strip())
                                last_events.append(obj)
                                if len(last_events) > 5:
                                    last_events.pop(0)
                            except json.JSONDecodeError:
                                pass
            except Exception:
                pass

        events_task = asyncio.create_task(consume_events())
        try:
            with Live(refresh_per_second=4, console=console, screen=False) as live:
                while True:
                    try:
                        deps_r = await c.get("/admin/deployments")
                        gpus_r = await c.get("/admin/gpus")
                        deps = deps_r.json() if deps_r.status_code == 200 else []
                        gpus = gpus_r.json() if gpus_r.status_code == 200 else []
                    except httpx.HTTPError as e:
                        console.print(f"daemon unreachable: {e}", style="red")
                        await asyncio.sleep(refresh_s)
                        continue

                    live.update(_render(deps, gpus, last_events))
                    await asyncio.sleep(refresh_s)
        finally:
            events_task.cancel()


def _render(deps: list[dict], gpus: list[dict], events: list[dict]):
    from rich.console import Group
    dep_table = Table(title="Deployments", show_lines=False)
    for col in ("ID", "STATUS", "PIN", "BACKEND", "GPUs", "VRAM(MB)", "CONTAINER"):
        dep_table.add_column(col)
    for d in deps:
        dep_table.add_row(
            str(d["id"]),
            d["status"],
            "*" if d.get("pinned") else "-",
            d["backend"],
            ",".join(str(g) for g in d.get("gpu_ids", [])),
            str(d.get("vram_reserved_mb", 0)),
            d.get("container_name") or "-",
        )
    gpu_table = Table(title="GPUs", show_lines=False)
    for col in ("INDEX", "MEM USED/TOTAL (MB)", "UTIL %", "POWER W"):
        gpu_table.add_column(col)
    for g in gpus:
        gpu_table.add_row(
            str(g["index"]),
            f"{g['memory_used_mb']}/{g['memory_total_mb']}",
            str(g["gpu_util_pct"]),
            str(g["power_w"]),
        )
    ev_table = Table(title="Recent events", show_lines=False)
    for col in ("TS", "KIND", "PAYLOAD"):
        ev_table.add_column(col)
    for e in events[-5:]:
        ts = e.get("ts", "")[-15:]
        ev_table.add_row(ts, e.get("kind", ""), json.dumps(e.get("payload", {})))
    return Group(dep_table, gpu_table, ev_table)
```

- [ ] **Step 2: Register in `cli/__init__.py`**

Add `top_cmd` to the import block (alphabetical position):
```python
    top_cmd,  # noqa: F401  registers command
```

- [ ] **Step 3: Verify import + commit**

```bash
python -c "from serve_engine.cli import top_cmd; print('ok')"
pytest -v
ruff check src/ tests/
git add src/serve_engine/cli/top_cmd.py src/serve_engine/cli/__init__.py
git commit -m "feat(cli): serve top - live htop-style dashboard"
```

---

## Task 7: Live verification

**Files:** none new - manual procedure documented at the end of this plan.

- [ ] **Step 1: With the daemon running and a deployment ready, verify each surface**

```bash
# In one terminal
serve daemon start
serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b
serve run qwen-0_5b --gpu 0

# /metrics
curl -s http://127.0.0.1:11500/metrics | head -20
# Should show serve_deployments, serve_models_total, serve_proxy_requests_total, plus vLLM's own metrics

# /admin/gpus (UDS only)
curl -s --unix-socket ~/.serve/sock http://localhost/admin/gpus

# /admin/events SSE - open in another terminal
curl -N --unix-socket ~/.serve/sock http://localhost/admin/events
# Then trigger an event: serve stop in a third terminal

# serve top
serve top
# Should show three tables updating every second
```

If all four surfaces produce output, Plan 05 is verified.

- [ ] **Step 2: Commit a CHANGELOG note** (optional)

```bash
# No code change; mark task complete in the tracker.
```

---

## Verification (end of Plan 05)

1. `pytest -v` - all tests pass.
2. `ruff check src/ tests/` - clean.
3. Live: `/metrics`, `/admin/gpus`, `/admin/events`, `serve top` all produce output.

## Self-review

- **Spec coverage:** event bus (T1), GPU snapshot (T2), Prometheus aggregation (T3), event emission + SSE (T4), `/admin/gpus` (T5), `serve top` (T6).
- **No autotune, no UI** - UI is Plan 06.
- **Backward compat:** EventBus param to LifecycleManager is optional; old tests that don't pass it still work.
- **Placeholder scan:** none.
- **Type consistency:** `Event`, `EventBus`, `GPUSnapshot` used identically.
