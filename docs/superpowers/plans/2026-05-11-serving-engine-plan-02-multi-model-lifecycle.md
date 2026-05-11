# Serving Engine — Plan 02: Multi-Model Lifecycle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift Plan 01's single-deployment restriction. The daemon now hosts N concurrent deployments, with explicit pin/auto-swap semantics, GPU-topology-aware placement, KV-cache-aware VRAM accounting, and LRU/idle eviction.

**Architecture:** Add a placement layer between the lifecycle manager and the Docker client. The manager no longer "stops the previous" — it asks placement for a GPU set, evicting auto deployments as needed. Routing changes from "find_active" to "find by model name." Pinned deployments are immune to eviction; auto deployments age out by idle timeout or get evicted to make room.

**Tech Stack:** Same as Plan 01 (Python 3.11+, FastAPI, Typer, sqlite3, docker-py, huggingface_hub) plus `pynvml` for GPU enumeration. No new third-party deps required for the placement / KV-cache math — pure Python.

**Explicitly NOT in this plan:** autotune (Plan 03), SGLang backend (Plan 04), API keys / fair queueing (Plan 05), metrics aggregation / `serve top` (Plan 06), web UI (Plan 07), doctor / install scripts (Plan 08).

---

## File structure produced by this plan

```
serving-engine/
├── src/serve_engine/
│   ├── store/
│   │   ├── deployments.py           # MODIFIED — new columns, new queries
│   │   ├── migrations/
│   │   │   └── 002_multi_deployment.sql   # NEW
│   ├── lifecycle/
│   │   ├── topology.py              # NEW — GPU enumeration & NVLink islands
│   │   ├── kv_estimator.py          # NEW — read config.json, compute KV bytes/token
│   │   ├── placement.py             # NEW — placement algorithm
│   │   ├── reaper.py                # NEW — idle-eviction background task
│   │   └── manager.py               # MODIFIED — N deployments, pin, eviction
│   ├── daemon/
│   │   ├── admin.py                 # MODIFIED — pin/unpin, ps with new columns
│   │   ├── app.py                   # MODIFIED — start reaper on startup
│   │   ├── __main__.py              # MODIFIED — pass topology to manager
│   │   └── openai_proxy.py          # MODIFIED — find_by_model_name, touch last_request_at
│   └── cli/
│       ├── pin_cmd.py               # NEW
│       ├── unpin_cmd.py             # NEW
│       ├── ps_cmd.py                # MODIFIED — new columns
│       ├── run_cmd.py               # MODIFIED — --pin and --idle-timeout flags
│       └── __init__.py              # MODIFIED — register new commands
└── tests/
    └── unit/
        ├── test_kv_estimator.py     # NEW
        ├── test_placement.py        # NEW
        ├── test_reaper.py           # NEW
        ├── test_lifecycle_manager.py  # MODIFIED — new scenarios
        └── test_admin_endpoints.py  # MODIFIED — pin/unpin/multi-deployment
```

---

## Task 1: Schema migration 002 (multi-deployment columns)

**Files:**
- Create: `src/serve_engine/store/migrations/002_multi_deployment.sql`
- Modify: `src/serve_engine/store/deployments.py`
- Modify: `tests/unit/test_store.py` (extend coverage)

- [ ] **Step 1: Write the migration SQL**

Create `src/serve_engine/store/migrations/002_multi_deployment.sql`:
```sql
-- Plan 02: multi-deployment lifecycle.
-- Add fields needed for pin/auto-swap, idle eviction, KV-aware placement,
-- and host-side routing addresses.

ALTER TABLE deployments ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
ALTER TABLE deployments ADD COLUMN idle_timeout_s INTEGER;
ALTER TABLE deployments ADD COLUMN vram_reserved_mb INTEGER NOT NULL DEFAULT 0;
ALTER TABLE deployments ADD COLUMN container_address TEXT;

CREATE INDEX IF NOT EXISTS idx_deployments_model_status ON deployments(model_id, status);
CREATE INDEX IF NOT EXISTS idx_deployments_last_request_at ON deployments(last_request_at);
```

(`last_request_at` already exists from the 001 schema; we just add an index on it for the reaper.)

- [ ] **Step 2: Extend the `Deployment` dataclass**

Modify `src/serve_engine/store/deployments.py`:

```python
@dataclass(frozen=True)
class Deployment:
    id: int
    model_id: int
    backend: str
    image_tag: str
    gpu_ids: list[int]
    tensor_parallel: int
    max_model_len: int | None
    dtype: str
    container_id: str | None
    container_name: str | None
    container_port: int | None
    container_address: str | None       # NEW
    status: Status
    last_error: str | None
    pinned: bool                        # NEW
    idle_timeout_s: int | None          # NEW
    vram_reserved_mb: int               # NEW
```

Update `_row_to_dep` to read all four new columns. Update `create()` to accept `pinned`, `idle_timeout_s`, `vram_reserved_mb` as keyword args (with defaults `pinned=False`, `idle_timeout_s=None`, `vram_reserved_mb=0`). Update `set_container` to also set `container_address`.

- [ ] **Step 3: Add new queries**

In `src/serve_engine/store/deployments.py`, add:

```python
def find_ready_by_model_name(conn: sqlite3.Connection, model_name: str) -> Deployment | None:
    """Return the most-recently-loaded ready deployment for a model, or None."""
    row = conn.execute(
        """
        SELECT d.* FROM deployments d
        JOIN models m ON m.id = d.model_id
        WHERE m.name = ? AND d.status = 'ready'
        ORDER BY d.started_at DESC LIMIT 1
        """,
        (model_name,),
    ).fetchone()
    return _row_to_dep(row) if row else None


def list_ready(conn: sqlite3.Connection) -> list[Deployment]:
    """All deployments currently in 'ready' status."""
    rows = conn.execute(
        "SELECT * FROM deployments WHERE status = 'ready' ORDER BY id"
    ).fetchall()
    return [_row_to_dep(r) for r in rows]


def list_evictable(conn: sqlite3.Connection) -> list[Deployment]:
    """Non-pinned ready deployments, sorted oldest-touched first (LRU)."""
    rows = conn.execute(
        """
        SELECT * FROM deployments
        WHERE status = 'ready' AND pinned = 0
        ORDER BY COALESCE(last_request_at, started_at) ASC
        """
    ).fetchall()
    return [_row_to_dep(r) for r in rows]


def touch_last_request(conn: sqlite3.Connection, dep_id: int) -> None:
    """Update last_request_at to now. Called by the proxy on every request."""
    conn.execute(
        "UPDATE deployments SET last_request_at = CURRENT_TIMESTAMP WHERE id = ?",
        (dep_id,),
    )


def set_pinned(conn: sqlite3.Connection, dep_id: int, pinned: bool) -> None:
    conn.execute(
        "UPDATE deployments SET pinned = ? WHERE id = ?",
        (1 if pinned else 0, dep_id),
    )
```

- [ ] **Step 4: Write tests for new queries**

Append to `tests/unit/test_store.py`:

```python
import time


def test_find_ready_by_model_name(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="qwen", hf_repo="org/qwen")
    assert dep_store.find_ready_by_model_name(conn, "qwen") is None
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=8192, dtype="auto",
    )
    dep_store.update_status(conn, d.id, "ready")
    dep_store.set_container(
        conn, d.id, container_id="cid", container_name="x",
        container_port=49152, container_address="127.0.0.1",
    )
    found = dep_store.find_ready_by_model_name(conn, "qwen")
    assert found is not None
    assert found.id == d.id
    assert found.container_address == "127.0.0.1"


def test_list_evictable_sorts_lru(tmp_path):
    conn = _fresh(tmp_path)
    m1 = model_store.add(conn, name="a", hf_repo="org/a")
    m2 = model_store.add(conn, name="b", hf_repo="org/b")
    m3 = model_store.add(conn, name="c", hf_repo="org/c")

    def _make(model_id: int, pinned: bool = False) -> int:
        d = dep_store.create(
            conn, model_id=model_id, backend="vllm", image_tag="img:v1",
            gpu_ids=[0], tensor_parallel=1, max_model_len=4096, dtype="auto",
            pinned=pinned,
        )
        dep_store.update_status(conn, d.id, "ready")
        return d.id

    da = _make(m1.id, pinned=True)
    db = _make(m2.id)
    dc = _make(m3.id)

    # Touch order: dc first (oldest), then db. Pinned da is excluded.
    dep_store.touch_last_request(conn, dc)
    time.sleep(0.01)
    dep_store.touch_last_request(conn, db)

    rows = dep_store.list_evictable(conn)
    assert [r.id for r in rows] == [dc, db]  # LRU: dc touched first → most-evictable
    assert da not in [r.id for r in rows]


def test_set_pinned(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=8192, dtype="auto",
    )
    assert dep_store.get_by_id(conn, d.id).pinned is False
    dep_store.set_pinned(conn, d.id, True)
    assert dep_store.get_by_id(conn, d.id).pinned is True
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_store.py -v
```
Expected: all existing tests pass + 3 new pass.

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/store/migrations/002_multi_deployment.sql src/serve_engine/store/deployments.py tests/unit/test_store.py
git commit -m "feat(store): schema 002 — pinned, idle_timeout, vram_reserved, container_address"
```

---

## Task 2: KV cache estimator

**Files:**
- Create: `src/serve_engine/lifecycle/kv_estimator.py`
- Create: `tests/unit/test_kv_estimator.py`

The estimator reads a model's `config.json` (already on disk after `serve pull`) and returns the **VRAM reservation in MB** for a given (max_model_len, target_concurrency, dtype).

Formula:
```
kv_bytes_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes
weights_bytes      = num_parameters * dtype_bytes
activation_factor  = 1.15   # observed empirical overhead for vLLM
total_bytes        = weights_bytes + kv_bytes_per_token * max_model_len * target_concurrency
total_mb           = ceil(total_bytes * activation_factor / 1024 / 1024)
```

Notes:
- `head_dim = hidden_size // num_attention_heads` if not explicit.
- `num_key_value_heads` defaults to `num_attention_heads` if missing (older models without GQA).
- `num_parameters` is read from `safetensors_total_size / dtype_bytes` if not present; otherwise estimated as `12 * num_hidden_layers * hidden_size**2` (rough but good enough for placement).
- `dtype_bytes`: bf16/fp16 → 2, fp8 → 1, auto → 2 (default to bf16).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_kv_estimator.py`:
```python
import json
import pytest

from serve_engine.lifecycle.kv_estimator import (
    estimate_vram_mb,
    KVEstimateInput,
    read_model_config,
)


def _write_config(tmp_path, **overrides):
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "torch_dtype": "bfloat16",
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return tmp_path


def test_estimate_basic(tmp_path):
    model_dir = _write_config(tmp_path)
    inp = KVEstimateInput(
        model_dir=model_dir,
        max_model_len=4096,
        target_concurrency=8,
        dtype="auto",
    )
    mb = estimate_vram_mb(inp)
    assert mb > 0
    # For 24 layers × 16 kv_heads × (1024/16=64) head_dim × 2 bytes
    # = 49,152 bytes per token. ×4096 ctx ×8 concurrency = ~1.5 GB just KV.
    assert 1000 < mb < 10000


def test_estimate_handles_gqa(tmp_path):
    # GQA: kv_heads < attention_heads → smaller KV
    full = _write_config(tmp_path)
    full_mb = estimate_vram_mb(
        KVEstimateInput(model_dir=full, max_model_len=4096,
                        target_concurrency=8, dtype="bf16")
    )

    gqa_dir = tmp_path / "gqa"
    gqa_dir.mkdir()
    _write_config(gqa_dir, num_key_value_heads=4)
    gqa_mb = estimate_vram_mb(
        KVEstimateInput(model_dir=gqa_dir, max_model_len=4096,
                        target_concurrency=8, dtype="bf16")
    )
    assert gqa_mb < full_mb


def test_dtype_fp8_halves_kv(tmp_path):
    md = _write_config(tmp_path)
    bf16 = estimate_vram_mb(KVEstimateInput(
        model_dir=md, max_model_len=4096, target_concurrency=8, dtype="bf16"))
    fp8 = estimate_vram_mb(KVEstimateInput(
        model_dir=md, max_model_len=4096, target_concurrency=8, dtype="fp8"))
    assert fp8 < bf16


def test_read_model_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_model_config(tmp_path)
```

- [ ] **Step 2: Run and confirm failure**

`pytest tests/unit/test_kv_estimator.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/kv_estimator.py`**

```python
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

ACTIVATION_OVERHEAD = 1.15


@dataclass(frozen=True)
class KVEstimateInput:
    model_dir: Path
    max_model_len: int
    target_concurrency: int
    dtype: str


def _dtype_bytes(dtype: str, torch_dtype: str | None) -> int:
    if dtype == "fp8":
        return 1
    if dtype in ("fp16", "bf16"):
        return 2
    if dtype == "auto":
        if torch_dtype in ("float16", "bfloat16"):
            return 2
        if torch_dtype == "float32":
            return 4
        return 2  # default
    return 2


def read_model_config(model_dir: Path) -> dict:
    p = Path(model_dir) / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"no config.json under {model_dir}")
    return json.loads(p.read_text())


def _estimate_param_bytes(cfg: dict, dtype_bytes: int) -> int:
    """Rough parameter count from config; used when safetensors metadata unavailable."""
    # 12 * L * H^2 covers attention + MLP for transformer roughly.
    L = int(cfg.get("num_hidden_layers", 0))
    H = int(cfg.get("hidden_size", 0))
    vocab = int(cfg.get("vocab_size", 0))
    if L == 0 or H == 0:
        return 0
    backbone = 12 * L * H * H
    embed = vocab * H * 2  # input + output embeddings
    return (backbone + embed) * dtype_bytes


def estimate_vram_mb(inp: KVEstimateInput) -> int:
    cfg = read_model_config(inp.model_dir)
    torch_dtype = cfg.get("torch_dtype")
    dtype_bytes = _dtype_bytes(inp.dtype, torch_dtype)

    n_layers = int(cfg.get("num_hidden_layers", 0))
    hidden = int(cfg.get("hidden_size", 0))
    n_heads = int(cfg.get("num_attention_heads", 1))
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", hidden // n_heads if n_heads else 0))

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
    kv_bytes = kv_bytes_per_token * inp.max_model_len * inp.target_concurrency
    weights_bytes = _estimate_param_bytes(cfg, dtype_bytes)

    total = (weights_bytes + kv_bytes) * ACTIVATION_OVERHEAD
    return math.ceil(total / 1024 / 1024)
```

- [ ] **Step 4: Run tests**

`pytest tests/unit/test_kv_estimator.py -v` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/lifecycle/kv_estimator.py tests/unit/test_kv_estimator.py
git commit -m "feat(lifecycle): KV-cache-aware VRAM estimator"
```

---

## Task 3: GPU topology reader

**Files:**
- Create: `src/serve_engine/lifecycle/topology.py`
- Create: `tests/unit/test_topology.py`

Read GPU count and per-GPU memory via `pynvml`. NVLink island detection is best-effort: `pynvml.nvmlDeviceGetNvLinkRemotePciInfo` per link, or parse `nvidia-smi topo -m` as fallback. Plan 02 uses the simpler approach: query each device's NVLink state; if any link to peer X is active, they're "connected".

- [ ] **Step 1: Write tests with pynvml mocked**

`tests/unit/test_topology.py`:
```python
from unittest.mock import MagicMock, patch

from serve_engine.lifecycle.topology import GPUInfo, Topology, read_topology


@patch("serve_engine.lifecycle.topology.pynvml")
def test_read_topology_basic(mock_nvml):
    mock_nvml.nvmlInit = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 2
    devs = [MagicMock(), MagicMock()]
    mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = devs
    mock_nvml.nvmlDeviceGetName.side_effect = [b"H100", b"H100"]
    mock_nvml.nvmlDeviceGetMemoryInfo.side_effect = [
        MagicMock(total=80 * 1024**3),
        MagicMock(total=80 * 1024**3),
    ]
    # NVLink active between 0 and 1
    mock_nvml.NVML_FI_DEV_NVLINK_LINK_COUNT = 1
    mock_nvml.nvmlDeviceGetFieldValues = MagicMock()
    # Pretend get_topology_common_ancestor returns NVLink for (0, 1)
    mock_nvml.nvmlDeviceGetTopologyCommonAncestor.return_value = 1  # NVLINK
    mock_nvml.NVML_TOPOLOGY_NVLINK = 1

    topo = read_topology()
    assert len(topo.gpus) == 2
    assert topo.gpus[0].total_mb == 80 * 1024
    # Both GPUs in the same island
    assert topo.nvlink_island(0) == frozenset({0, 1})


@patch("serve_engine.lifecycle.topology.pynvml")
def test_read_topology_no_nvlink(mock_nvml):
    mock_nvml.nvmlInit = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
    mock_nvml.nvmlDeviceGetName.return_value = b"A100"
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(total=40 * 1024**3)

    topo = read_topology()
    assert len(topo.gpus) == 1
    assert topo.nvlink_island(0) == frozenset({0})  # singleton island
```

- [ ] **Step 2: Run, confirm failure**

`pytest tests/unit/test_topology.py -v` → FAIL.

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/topology.py`**

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from functools import cache

try:
    import pynvml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mb: int


@dataclass(frozen=True)
class Topology:
    gpus: list[GPUInfo]
    # Map gpu_index -> frozenset of NVLink-peer indices (including self).
    _islands: dict[int, frozenset[int]] = field(default_factory=dict)

    def nvlink_island(self, index: int) -> frozenset[int]:
        return self._islands.get(index, frozenset({index}))


def _build_islands(count: int) -> dict[int, frozenset[int]]:
    """Group GPUs into NVLink-connected sets via union-find."""
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    if pynvml is None:
        return {i: frozenset({i}) for i in range(count)}

    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            try:
                rel = pynvml.nvmlDeviceGetTopologyCommonAncestor(handles[i], handles[j])
            except Exception:
                continue
            if rel == pynvml.NVML_TOPOLOGY_NVLINK:
                union(i, j)

    islands: dict[int, set[int]] = {}
    for i in range(count):
        islands.setdefault(find(i), set()).add(i)
    out: dict[int, frozenset[int]] = {}
    for members in islands.values():
        s = frozenset(members)
        for m in members:
            out[m] = s
    return out


@cache
def read_topology() -> Topology:
    """Enumerate GPUs and detect NVLink islands. Cached for the process lifetime."""
    if pynvml is None:
        log.warning("pynvml unavailable — no GPUs visible")
        return Topology(gpus=[], _islands={})

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    gpus: list[GPUInfo] = []
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name_raw = pynvml.nvmlDeviceGetName(h)
        name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        gpus.append(GPUInfo(index=i, name=name, total_mb=int(mem.total) // 1024 // 1024))
    islands = _build_islands(count)
    return Topology(gpus=gpus, _islands=islands)


def reset_cache() -> None:
    """For tests / re-detection after driver hot-plug."""
    read_topology.cache_clear()
```

- [ ] **Step 4: Add pynvml dependency**

Edit `pyproject.toml`, add to `[project] dependencies`:
```
"pynvml>=11.5",
```

Run `uv pip install -e ".[dev]"` to install.

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_topology.py -v
```
Expected: 2 pass.

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/lifecycle/topology.py tests/unit/test_topology.py pyproject.toml
git commit -m "feat(lifecycle): GPU topology reader with NVLink island detection"
```

---

## Task 4: Placement algorithm

**Files:**
- Create: `src/serve_engine/lifecycle/placement.py`
- Create: `tests/unit/test_placement.py`

Given (topology, currently-allocated deployments, requested deployment shape with `vram_reserved_mb`), return either:
- A `Fit` with the GPU IDs to use, OR
- An `EvictThenFit` listing which auto deployments to evict to make room, OR
- A `NoRoom` error.

Algorithm:
1. Compute `available_mb[gpu_id] = total_mb − sum(reserved_mb of allocated deployments on this gpu)`.
2. Find an NVLink-island-aligned subset of free GPUs of size `tp`. If found, return `Fit`.
3. Otherwise, consider evictions: walk `list_evictable` (LRU order) and "subtract" their reservations. After each subtraction, retry step 2. First success wins.
4. If no eviction sequence works, return `NoRoom`.

- [ ] **Step 1: Tests**

`tests/unit/test_placement.py`:
```python
from dataclasses import replace

from serve_engine.lifecycle.placement import (
    AllocatedDeployment,
    Fit,
    EvictThenFit,
    NoRoom,
    PlacementRequest,
    plan_placement,
)
from serve_engine.lifecycle.topology import GPUInfo, Topology


def _topo(n: int, nvlink: bool = True) -> Topology:
    gpus = [GPUInfo(index=i, name="H100", total_mb=80 * 1024) for i in range(n)]
    if nvlink and n > 1:
        islands = {i: frozenset(range(n)) for i in range(n)}
    else:
        islands = {i: frozenset({i}) for i in range(n)}
    return Topology(gpus=gpus, _islands=islands)


def test_fit_on_free_gpu():
    req = PlacementRequest(
        tensor_parallel=1, vram_reserved_mb=20_000, model_name="x",
    )
    decision = plan_placement(_topo(2), allocated=[], request=req)
    assert isinstance(decision, Fit)
    assert decision.gpu_ids == [0]


def test_fit_tp2_on_nvlink_pair():
    req = PlacementRequest(
        tensor_parallel=2, vram_reserved_mb=70_000, model_name="x",
    )
    decision = plan_placement(_topo(4), allocated=[], request=req)
    assert isinstance(decision, Fit)
    assert len(decision.gpu_ids) == 2


def test_no_room_when_total_vram_insufficient():
    req = PlacementRequest(
        tensor_parallel=1, vram_reserved_mb=200_000, model_name="x",
    )
    decision = plan_placement(_topo(1), allocated=[], request=req)
    assert isinstance(decision, NoRoom)


def test_evict_then_fit_lru_first():
    topo = _topo(2)
    # GPU 0 occupied by an auto deployment with 70GB reserved
    alloc = [
        AllocatedDeployment(id=1, gpu_ids=[0], vram_reserved_mb=70_000, pinned=False),
    ]
    req = PlacementRequest(
        tensor_parallel=1, vram_reserved_mb=70_000, model_name="x",
    )
    # GPU 1 is free → fit there directly
    decision = plan_placement(topo, allocated=alloc, request=req)
    assert isinstance(decision, Fit)
    assert decision.gpu_ids == [1]

    # Now both GPUs are full
    alloc2 = alloc + [
        AllocatedDeployment(id=2, gpu_ids=[1], vram_reserved_mb=70_000, pinned=False),
    ]
    decision = plan_placement(topo, allocated=alloc2, request=req)
    assert isinstance(decision, EvictThenFit)
    # LRU comes first: id=1 was added first → it's the LRU here (caller orders list)
    assert decision.evict_ids == [1]
    assert decision.gpu_ids == [0]


def test_pinned_blocks_eviction():
    topo = _topo(1)
    alloc = [
        AllocatedDeployment(id=1, gpu_ids=[0], vram_reserved_mb=70_000, pinned=True),
    ]
    req = PlacementRequest(
        tensor_parallel=1, vram_reserved_mb=70_000, model_name="x",
    )
    decision = plan_placement(topo, allocated=alloc, request=req)
    assert isinstance(decision, NoRoom)


def test_tp_must_be_power_of_two_or_match_islands():
    topo = _topo(4)
    req = PlacementRequest(
        tensor_parallel=3, vram_reserved_mb=20_000, model_name="x",
    )
    decision = plan_placement(topo, allocated=[], request=req)
    assert isinstance(decision, NoRoom)
    assert "power of 2" in decision.reason.lower()
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/placement.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

from serve_engine.lifecycle.topology import Topology


@dataclass(frozen=True)
class AllocatedDeployment:
    id: int
    gpu_ids: list[int]
    vram_reserved_mb: int
    pinned: bool


@dataclass(frozen=True)
class PlacementRequest:
    tensor_parallel: int
    vram_reserved_mb: int
    model_name: str


@dataclass(frozen=True)
class Fit:
    gpu_ids: list[int]


@dataclass(frozen=True)
class EvictThenFit:
    evict_ids: list[int]
    gpu_ids: list[int]


@dataclass(frozen=True)
class NoRoom:
    reason: str


Decision = Fit | EvictThenFit | NoRoom


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _available_mb(topo: Topology, allocated: list[AllocatedDeployment]) -> dict[int, int]:
    avail = {g.index: g.total_mb for g in topo.gpus}
    for a in allocated:
        share = a.vram_reserved_mb // len(a.gpu_ids)
        for g in a.gpu_ids:
            avail[g] = max(0, avail.get(g, 0) - share)
    return avail


def _try_fit(
    topo: Topology,
    avail: dict[int, int],
    req: PlacementRequest,
) -> list[int] | None:
    if not _is_power_of_two(req.tensor_parallel):
        return None
    per_gpu = req.vram_reserved_mb // req.tensor_parallel

    # Single-GPU case: any free GPU works.
    if req.tensor_parallel == 1:
        for g in sorted(avail, key=lambda i: -avail[i]):
            if avail[g] >= per_gpu:
                return [g]
        return None

    # TP > 1: need NVLink-connected GPUs.
    for island_seed in avail:
        island = topo.nvlink_island(island_seed)
        candidates = [g for g in sorted(island) if avail.get(g, 0) >= per_gpu]
        if len(candidates) < req.tensor_parallel:
            continue
        # Pick the first contiguous combination that fits.
        for combo in combinations(candidates, req.tensor_parallel):
            return list(combo)
    return None


def plan_placement(
    topo: Topology,
    *,
    allocated: list[AllocatedDeployment],
    request: PlacementRequest,
) -> Decision:
    if not _is_power_of_two(request.tensor_parallel):
        return NoRoom(reason=f"tensor_parallel={request.tensor_parallel} is not a power of 2")
    if request.tensor_parallel > len(topo.gpus):
        return NoRoom(reason=f"tensor_parallel={request.tensor_parallel} exceeds GPU count {len(topo.gpus)}")

    avail = _available_mb(topo, allocated)
    fit = _try_fit(topo, avail, request)
    if fit is not None:
        return Fit(gpu_ids=fit)

    # Try evicting auto (non-pinned) deployments in the order given (LRU first).
    evictable = [a for a in allocated if not a.pinned]
    evicted_ids: list[int] = []
    for victim in evictable:
        # Apply eviction
        share = victim.vram_reserved_mb // len(victim.gpu_ids)
        for g in victim.gpu_ids:
            avail[g] = min(
                topo.gpus[g].total_mb if g < len(topo.gpus) else 0,
                avail.get(g, 0) + share,
            )
        evicted_ids.append(victim.id)
        fit = _try_fit(topo, avail, request)
        if fit is not None:
            return EvictThenFit(evict_ids=evicted_ids, gpu_ids=fit)

    return NoRoom(
        reason=(
            f"cannot place {request.model_name!r}: needs "
            f"{request.vram_reserved_mb} MB across {request.tensor_parallel} GPUs; "
            "no fit even after evicting all auto deployments"
        )
    )
```

- [ ] **Step 4: Run tests, all pass.**

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/lifecycle/placement.py tests/unit/test_placement.py
git commit -m "feat(lifecycle): topology-aware placement with LRU eviction"
```

---

## Task 5: LifecycleManager refactor — multi-deployment

**Files:**
- Modify: `src/serve_engine/lifecycle/manager.py`
- Modify: `tests/unit/test_lifecycle_manager.py`

The manager now:
1. Does NOT stop the previous deployment automatically on `load()`. It calls `plan_placement` and evicts only what's needed.
2. Accepts `pinned: bool` and `idle_timeout_s: int | None` on the plan.
3. Persists `vram_reserved_mb` (via KV estimator) and `container_address` (from `ContainerHandle`).
4. Exposes a public `docker_client` accessor (`mgr.docker` property).
5. Adds a `pin(dep_id, pinned: bool)` method.
6. `stop(dep_id: int | None)` — stops a specific deployment, or all if `dep_id is None`.

**Important:** `DeploymentPlan` gains two new optional fields (`pinned`, `idle_timeout_s`). Keep backward compat: defaults `pinned=False`, `idle_timeout_s=None`.

- [ ] **Step 1: Extend `DeploymentPlan`**

In `src/serve_engine/lifecycle/plan.py`, add fields after `extra_args`:

```python
    pinned: bool = False
    idle_timeout_s: int | None = None
    target_concurrency: int = 8  # used by KV estimator
```

`target_concurrency` is what the KV estimator uses to size the budget.

- [ ] **Step 2: Rewrite `LifecycleManager.load`**

The full new `manager.py` (replace the existing `LifecycleManager` class — keep the module-level `wait_healthy` and `download_model_async` functions as they are):

```python
class LifecycleManager:
    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        docker_client: DockerClient,
        backends: dict[str, Backend],
        models_dir: Path,
        topology: Topology | None = None,
        load_timeout_s: float = 600.0,
    ):
        self._conn = conn
        self._docker = docker_client
        self._backends = backends
        self._models_dir = models_dir
        self._topology = topology
        self._load_timeout_s = load_timeout_s
        self._lock = asyncio.Lock()

    @property
    def docker(self) -> DockerClient:
        return self._docker

    async def load(self, plan: DeploymentPlan):
        async with self._lock:
            # 1. Ensure model row exists
            model = model_store.get_by_name(self._conn, plan.model_name)
            if model is None:
                model = model_store.add(
                    self._conn,
                    name=plan.model_name,
                    hf_repo=plan.hf_repo,
                    revision=plan.revision,
                )

            # 2. Ensure weights are local
            local_path = model.local_path
            if local_path is None:
                local_path = await download_model_async(
                    hf_repo=plan.hf_repo,
                    revision=plan.revision,
                    cache_dir=self._models_dir,
                )
                model_store.set_local_path(self._conn, model.id, local_path)

            # 3. Estimate VRAM
            from serve_engine.lifecycle.kv_estimator import (
                KVEstimateInput, estimate_vram_mb,
            )
            vram_mb = estimate_vram_mb(KVEstimateInput(
                model_dir=Path(local_path),
                max_model_len=plan.max_model_len,
                target_concurrency=plan.target_concurrency,
                dtype=plan.dtype,
            ))

            # 4. Placement
            from serve_engine.lifecycle.placement import (
                AllocatedDeployment, PlacementRequest, plan_placement,
                Fit, EvictThenFit, NoRoom,
            )
            ready = dep_store.list_ready(self._conn)
            evictable_order = {d.id: idx for idx, d in enumerate(dep_store.list_evictable(self._conn))}
            allocated = sorted(
                [
                    AllocatedDeployment(
                        id=d.id,
                        gpu_ids=d.gpu_ids,
                        vram_reserved_mb=d.vram_reserved_mb,
                        pinned=d.pinned,
                    )
                    for d in ready
                ],
                key=lambda a: (a.pinned, evictable_order.get(a.id, -1)),
            )
            request = PlacementRequest(
                tensor_parallel=plan.tensor_parallel,
                vram_reserved_mb=vram_mb,
                model_name=plan.model_name,
            )
            if self._topology is None:
                raise RuntimeError("topology not initialized; pass topology=read_topology() to LifecycleManager")
            decision = plan_placement(self._topology, allocated=allocated, request=request)

            if isinstance(decision, NoRoom):
                raise RuntimeError(decision.reason)
            if isinstance(decision, EvictThenFit):
                for victim_id in decision.evict_ids:
                    await self._stop_locked(victim_id)
                gpu_ids = decision.gpu_ids
            else:  # Fit
                gpu_ids = decision.gpu_ids

            # 5. Create row + spawn container (gpu_ids may differ from plan.gpu_ids)
            dep = dep_store.create(
                self._conn,
                model_id=model.id,
                backend=plan.backend,
                image_tag=plan.image_tag,
                gpu_ids=gpu_ids,
                tensor_parallel=plan.tensor_parallel,
                max_model_len=plan.max_model_len,
                dtype=plan.dtype,
                pinned=plan.pinned,
                idle_timeout_s=plan.idle_timeout_s,
                vram_reserved_mb=vram_mb,
            )
            dep_store.update_status(self._conn, dep.id, "loading")

            backend = self._backends[plan.backend]
            container_model_path = "/cache/" + str(
                Path(local_path).resolve().relative_to(self._models_dir.resolve())
            )
            # Rebuild plan with the placement-chosen GPU set so backend argv reflects reality
            effective_plan = replace(
                plan,
                gpu_ids=list(gpu_ids),
                tensor_parallel=len(gpu_ids),
            )
            handle = self._docker.run(
                image=plan.image_tag,
                name=f"serve-{plan.backend}-{plan.model_name}-{dep.id}",
                command=backend.build_argv(effective_plan, local_model_path=container_model_path),
                environment=backend.container_env(effective_plan),
                kwargs=backend.container_kwargs(effective_plan),
                volumes={str(self._models_dir.resolve()): {"bind": "/cache", "mode": "ro"}},
                internal_port=8000,
            )
            dep_store.set_container(
                self._conn, dep.id,
                container_id=handle.id,
                container_name=handle.name,
                container_port=handle.port,
                container_address=handle.address,
            )

            health_url = f"http://{handle.address}:{handle.port}{backend.health_path}"
            ok = await wait_healthy(health_url, timeout_s=self._load_timeout_s)
            if not ok:
                self._docker.stop(handle.id, timeout=10)
                msg = f"engine did not become healthy within load timeout ({health_url})"
                dep_store.update_status(self._conn, dep.id, "failed", last_error=msg)
                raise RuntimeError(msg)

            dep_store.update_status(self._conn, dep.id, "ready")
            return dep_store.get_by_id(self._conn, dep.id)

    async def stop(self, dep_id: int | None = None) -> None:
        async with self._lock:
            if dep_id is None:
                ready = dep_store.list_ready(self._conn)
                for d in ready:
                    await self._stop_locked(d.id)
            else:
                await self._stop_locked(dep_id)

    async def pin(self, dep_id: int, pinned: bool = True) -> None:
        dep_store.set_pinned(self._conn, dep_id, pinned)

    async def _stop_locked(self, dep_id: int) -> None:
        dep = dep_store.get_by_id(self._conn, dep_id)
        if dep is None:
            return
        dep_store.update_status(self._conn, dep.id, "stopping")
        if dep.container_id:
            self._docker.stop(dep.container_id, timeout=30)
        dep_store.update_status(self._conn, dep.id, "stopped")
```

Add imports to the top of the file:
```python
from dataclasses import replace
from serve_engine.lifecycle.topology import Topology
```

The `set_container` call now passes `container_address=handle.address` — that requires updating `dep_store.set_container` (done in Task 1).

- [ ] **Step 3: Update `set_container` signature**

In `src/serve_engine/store/deployments.py`, change `set_container` to:
```python
def set_container(
    conn: sqlite3.Connection,
    dep_id: int,
    *,
    container_id: str,
    container_name: str,
    container_port: int,
    container_address: str,
) -> None:
    conn.execute(
        """
        UPDATE deployments
        SET container_id=?, container_name=?, container_port=?, container_address=?,
            started_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (container_id, container_name, container_port, container_address, dep_id),
    )
```

- [ ] **Step 4: Update existing test for `set_container` in `tests/unit/test_store.py`**

In `test_set_container_info`, pass the new `container_address="172.18.0.5"` kwarg and assert it round-trips.

- [ ] **Step 5: Update existing manager tests**

In `tests/unit/test_lifecycle_manager.py`:
- Inject a topology fixture (`Topology(gpus=[GPUInfo(0, "H100", 80_000)], _islands={0: frozenset({0})})`).
- Pass `topology=...` to every `LifecycleManager` constructor call.
- Patch `serve_engine.lifecycle.manager.estimate_vram_mb` to return a fixed value (e.g. 20_000) so tests don't depend on disk config.json files.
- `test_load_stops_previous_deployment` becomes `test_load_evicts_previous_when_room_constrained` — set the topology to have only one GPU, the first load occupies it, the second forces eviction.

Concrete test additions:

```python
@pytest.fixture
def topo_one_gpu():
    from serve_engine.lifecycle.topology import Topology, GPUInfo
    return Topology(
        gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024)],
        _islands={0: frozenset({0})},
    )


def test_load_starts_engine_and_marks_ready(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="vllm-llama-1b", address="127.0.0.1", port=49152,
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.estimate_vram_mb",
        lambda inp: 20_000,
    )
    (tmp_path / "weights").mkdir()
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    dep = asyncio.run(mgr.load(_make_plan()))
    assert dep.status == "ready"
    assert dep.container_address == "127.0.0.1"
    assert dep.vram_reserved_mb == 20_000


def test_load_evicts_previous_when_room_constrained(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.side_effect = [
        ContainerHandle(id="cid1", name="x1", address="127.0.0.1", port=49152),
        ContainerHandle(id="cid2", name="x2", address="127.0.0.1", port=49153),
    ]
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )
    # 60 GB per deployment, 80 GB total → can only fit one at a time
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.estimate_vram_mb",
        lambda inp: 60 * 1024,
    )
    (tmp_path / "weights").mkdir()
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(_make_plan()))
    asyncio.run(mgr.load(_make_plan()))
    docker_client.stop.assert_called_once_with("cid1", timeout=30)


def test_pin_prevents_eviction(conn, monkeypatch, tmp_path, topo_one_gpu):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid1", name="x1", address="127.0.0.1", port=49152,
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.estimate_vram_mb",
        lambda inp: 60 * 1024,
    )
    (tmp_path / "weights").mkdir()
    mgr = LifecycleManager(
        conn=conn, docker_client=docker_client,
        backends={"vllm": VLLMBackend()}, models_dir=tmp_path,
        topology=topo_one_gpu,
    )
    plan = _make_plan()
    plan_pinned = replace(plan, pinned=True)
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(plan_pinned))
    # Loading another should fail with NoRoom
    plan2 = _make_plan()
    with pytest.raises(RuntimeError, match="cannot place"):
        asyncio.run(mgr.load(plan2))
```

Also import `from dataclasses import replace` at the top of the test file.

- [ ] **Step 6: Run all tests**

```bash
pytest -v
```
Expected: all pass (existing 37 + 3 new + Task-1's 3 + Task-2's 4 + Task-3's 2 + Task-4's 6).

- [ ] **Step 7: Commit**

```bash
git add src/serve_engine/lifecycle/manager.py src/serve_engine/lifecycle/plan.py src/serve_engine/store/deployments.py tests/unit/test_lifecycle_manager.py tests/unit/test_store.py
git commit -m "feat(lifecycle): multi-deployment manager with KV-aware placement and pin"
```

---

## Task 6: Wire topology + manager change into daemon entry point

**Files:**
- Modify: `src/serve_engine/daemon/__main__.py`
- Modify: `src/serve_engine/daemon/app.py`
- Modify: `tests/unit/test_admin_endpoints.py`

- [ ] **Step 1: Pass topology into `build_apps`**

In `src/serve_engine/daemon/app.py`, change `build_apps` and `build_app` to accept an optional `topology: Topology | None = None` parameter, and pass it through to `LifecycleManager(...)`.

Add at the top:
```python
from serve_engine.lifecycle.topology import Topology
```

In both `build_apps` and `build_app`, add `topology: Topology | None = None` to the signature and pass `topology=topology` to the `LifecycleManager` constructor inside.

- [ ] **Step 2: Call `read_topology()` in `__main__.py`**

In `src/serve_engine/daemon/__main__.py`, change:
```python
    tcp_app, uds_app = build_apps(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=config.MODELS_DIR,
    )
```
to:
```python
    from serve_engine.lifecycle.topology import read_topology
    topology = read_topology()
    log.info("topology: %d GPUs, islands=%s",
             len(topology.gpus),
             [list(topology.nvlink_island(g.index)) for g in topology.gpus])
    tcp_app, uds_app = build_apps(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=config.MODELS_DIR,
        topology=topology,
    )
```

(`log` is already imported / use `logging.getLogger(__name__)`.)

- [ ] **Step 3: Update `tests/unit/test_admin_endpoints.py`**

In the `app` fixture, build a fake one-GPU topology and pass it to `build_app`:

```python
from serve_engine.lifecycle.topology import Topology, GPUInfo

# Inside the fixture, before build_app:
topo = Topology(
    gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024)],
    _islands={0: frozenset({0})},
)
# Patch estimate_vram_mb so admin tests don't need a real config.json
monkeypatch.setattr(
    "serve_engine.lifecycle.manager.estimate_vram_mb",
    lambda inp: 20_000,
)
# Make models_dir exist as a fake snapshot tree
(tmp_path / "weights").mkdir(exist_ok=True)
app = build_app(
    conn=conn,
    docker_client=docker_client,
    backends={"vllm": VLLMBackend()},
    models_dir=tmp_path,
    topology=topo,
)
```

- [ ] **Step 4: Run tests, all pass**

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/daemon/__main__.py src/serve_engine/daemon/app.py tests/unit/test_admin_endpoints.py
git commit -m "feat(daemon): wire GPU topology into app + manager startup"
```

---

## Task 7: Proxy routes by model name + touches last_request_at

**Files:**
- Modify: `src/serve_engine/daemon/openai_proxy.py`
- Modify: `tests/integration/test_openai_proxy.py`

The proxy now:
1. Reads `model` from the incoming JSON body.
2. Looks up that model's most-recently-loaded ready deployment.
3. Updates `last_request_at` before forwarding.
4. Uses the deployment's `container_address` (not hardcoded 127.0.0.1).

- [ ] **Step 1: Implement the new proxy**

Replace the body of `_proxy` in `src/serve_engine/daemon/openai_proxy.py`:

```python
async def _proxy(request: Request, openai_subpath: str) -> StreamingResponse:
    conn: sqlite3.Connection = request.app.state.conn
    backends: dict[str, Backend] = request.app.state.backends

    body = await request.body()
    model_name: str | None = None
    try:
        parsed = json.loads(body) if body else {}
        if isinstance(parsed, dict):
            model_name = parsed.get("model")
    except json.JSONDecodeError:
        pass

    if not model_name:
        raise HTTPException(400, detail="request body must include 'model'")

    active = dep_store.find_ready_by_model_name(conn, model_name)
    if active is None or active.container_address is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"no ready deployment for model {model_name!r}",
        )

    backend = backends.get(active.backend)
    if backend is None:
        raise HTTPException(500, detail=f"unknown backend {active.backend!r}")

    dep_store.touch_last_request(conn, active.id)

    base = f"http://{active.container_address}:{active.container_port}{backend.openai_base}"
    _HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }

    client = make_engine_client(base)
    upstream = client.stream("POST", openai_subpath, content=body, headers=headers)

    async def streamer():
        try:
            async with upstream as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(streamer(), media_type="text/event-stream")
```

Add `import json` at the top of the file.

- [ ] **Step 2: Update `GET /v1/models`**

```python
@router.get("/v1/models")
def models(request: Request):
    conn: sqlite3.Connection = request.app.state.conn
    ready_by_model: dict[int, dep_store.Deployment] = {}
    for d in dep_store.list_ready(conn):
        ready_by_model[d.model_id] = d
    rows = model_store.list_all(conn)
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "owned_by": "serve-engine",
                "loaded": m.id in ready_by_model,
                "pinned": ready_by_model[m.id].pinned if m.id in ready_by_model else False,
            }
            for m in rows
        ],
    }
```

- [ ] **Step 3: Update integration test**

In `tests/integration/test_openai_proxy.py`, the existing `test_proxy_streams_response` should still pass — the `model` field is already in the request body. But the deployment in the fixture must have `container_address="127.0.0.1"`. The admin POST in the fixture should already set that via the manager's `set_container` call after Task 5's change.

Add a new test:
```python
@pytest.mark.asyncio
async def test_proxy_routes_by_model_name(app_with_active_deployment):
    app, _ = app_with_active_deployment
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "no-such-model", "messages": []},
        )
    assert r.status_code == 503
    assert "no ready deployment" in r.json()["detail"]


@pytest.mark.asyncio
async def test_proxy_400_when_no_model_field(app_with_active_deployment):
    app, _ = app_with_active_deployment
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
    assert r.status_code == 400
```

- [ ] **Step 4: Run, all pass**

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/daemon/openai_proxy.py tests/integration/test_openai_proxy.py
git commit -m "feat(daemon): proxy routes by model name and touches last_request_at"
```

---

## Task 8: Idle reaper background task

**Files:**
- Create: `src/serve_engine/lifecycle/reaper.py`
- Create: `tests/unit/test_reaper.py`
- Modify: `src/serve_engine/daemon/app.py`

The reaper runs every `tick_s` seconds. For each ready, non-pinned deployment with `idle_timeout_s` exceeded, call `manager.stop(dep_id)`.

Default tick = 30 s. Default idle timeout per deployment = 300 s (used when `idle_timeout_s` is NULL).

- [ ] **Step 1: Test**

`tests/unit/test_reaper.py`:
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from serve_engine.lifecycle.reaper import Reaper


@pytest.mark.asyncio
async def test_reaper_evicts_idle():
    now = 1_000_000
    deployments = [
        # idle 600s; default timeout 300 → evict
        MagicMock(id=1, pinned=False, idle_timeout_s=None,
                  last_request_at=now - 600, status="ready"),
        # idle 100s; default timeout 300 → keep
        MagicMock(id=2, pinned=False, idle_timeout_s=None,
                  last_request_at=now - 100, status="ready"),
        # pinned → keep regardless
        MagicMock(id=3, pinned=True, idle_timeout_s=None,
                  last_request_at=now - 10_000, status="ready"),
    ]
    manager = MagicMock()
    manager.stop = AsyncMock()

    list_ready = MagicMock(return_value=deployments)

    reaper = Reaper(
        manager=manager,
        list_ready=list_ready,
        default_idle_timeout_s=300,
        now_fn=lambda: now,
    )
    await reaper.tick_once()

    manager.stop.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_reaper_respects_per_deployment_timeout():
    now = 1_000_000
    deployments = [
        # idle 100s; per-deployment 60 → evict
        MagicMock(id=1, pinned=False, idle_timeout_s=60,
                  last_request_at=now - 100, status="ready"),
        # idle 100s; per-deployment 600 → keep
        MagicMock(id=2, pinned=False, idle_timeout_s=600,
                  last_request_at=now - 100, status="ready"),
    ]
    manager = MagicMock()
    manager.stop = AsyncMock()
    reaper = Reaper(
        manager=manager,
        list_ready=MagicMock(return_value=deployments),
        default_idle_timeout_s=300,
        now_fn=lambda: now,
    )
    await reaper.tick_once()
    manager.stop.assert_called_once_with(1)
```

- [ ] **Step 2: Implement `src/serve_engine/lifecycle/reaper.py`**

```python
from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable

log = logging.getLogger(__name__)


class Reaper:
    """Periodically evict ready non-pinned deployments past their idle timeout."""

    def __init__(
        self,
        *,
        manager,
        list_ready: Callable,
        default_idle_timeout_s: int = 300,
        tick_s: float = 30.0,
        now_fn: Callable[[], float] = time.time,
    ):
        self._manager = manager
        self._list_ready = list_ready
        self._default_idle_timeout_s = default_idle_timeout_s
        self._tick_s = tick_s
        self._now = now_fn
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def tick_once(self) -> None:
        now = self._now()
        for d in self._list_ready():
            if d.pinned:
                continue
            if d.status != "ready":
                continue
            last = d.last_request_at
            if last is None:
                continue
            try:
                last_ts = (
                    last if isinstance(last, (int, float))
                    else _parse_sqlite_ts(last)
                )
            except Exception:
                continue
            idle = now - last_ts
            timeout = d.idle_timeout_s or self._default_idle_timeout_s
            if idle >= timeout:
                log.info("reaper evicting deployment %s (idle %.0fs)", d.id, idle)
                try:
                    await self._manager.stop(d.id)
                except Exception:
                    log.exception("reaper failed to evict %s", d.id)

    async def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.tick_once()
            except Exception:
                log.exception("reaper tick failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._tick_s)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None


def _parse_sqlite_ts(s: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' (UTC) returned by SQLite CURRENT_TIMESTAMP."""
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
```

- [ ] **Step 3: Hook reaper into `app.py`**

In `src/serve_engine/daemon/app.py`, add a startup event to `build_apps` that starts the reaper on the UDS app's `LifecycleManager`. Since both apps share the same manager, hooking it once is enough. After:

```python
manager = LifecycleManager(...)
```

add:

```python
    from serve_engine.lifecycle.reaper import Reaper
    from serve_engine.store import deployments as _dep_store
    reaper = Reaper(
        manager=manager,
        list_ready=lambda: _dep_store.list_ready(conn),
    )
```

And add a startup hook on `uds_app` only (so the TCP app doesn't double-start):
```python
    @uds_app.on_event("startup")
    async def _start_reaper():
        reaper.start()

    @uds_app.on_event("shutdown")
    async def _stop_reaper():
        await reaper.stop()
```

- [ ] **Step 4: Run tests, all pass.**

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/lifecycle/reaper.py src/serve_engine/daemon/app.py tests/unit/test_reaper.py
git commit -m "feat(lifecycle): idle-eviction reaper"
```

---

## Task 9: Admin pin/unpin endpoints

**Files:**
- Modify: `src/serve_engine/daemon/admin.py`
- Modify: `tests/unit/test_admin_endpoints.py`

- [ ] **Step 1: Add endpoints**

Append to `src/serve_engine/daemon/admin.py`:

```python
@router.post("/deployments/{dep_id}/pin", status_code=status.HTTP_204_NO_CONTENT)
async def pin_deployment(
    dep_id: int,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if dep_store.get_by_id(conn, dep_id) is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    await manager.pin(dep_id, True)


@router.post("/deployments/{dep_id}/unpin", status_code=status.HTTP_204_NO_CONTENT)
async def unpin_deployment(
    dep_id: int,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if dep_store.get_by_id(conn, dep_id) is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    await manager.pin(dep_id, False)
```

- [ ] **Step 2: Allow `pinned` and `idle_timeout_s` in `CreateDeploymentRequest`**

Edit `CreateDeploymentRequest` in `admin.py`:
```python
class CreateDeploymentRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    hf_repo: str
    revision: str = "main"
    backend: str = "vllm"
    image_tag: str | None = None
    gpu_ids: list[int]
    tensor_parallel: int | None = None
    max_model_len: int = 8192
    dtype: str = "auto"
    pinned: bool = False
    idle_timeout_s: int | None = None
    target_concurrency: int = 8
```

And in `create_deployment`, pass them through:
```python
        plan = DeploymentPlan(
            model_name=body.model_name,
            hf_repo=body.hf_repo,
            revision=body.revision,
            backend=body.backend,
            image_tag=image_tag,
            gpu_ids=body.gpu_ids,
            tensor_parallel=tp,
            max_model_len=body.max_model_len,
            dtype=body.dtype,
            pinned=body.pinned,
            idle_timeout_s=body.idle_timeout_s,
            target_concurrency=body.target_concurrency,
        )
```

- [ ] **Step 3: Update `DELETE /admin/deployments/current` to accept an optional id**

Replace:
```python
@router.delete("/deployments/current", status_code=status.HTTP_204_NO_CONTENT)
async def stop_current(manager: LifecycleManager = Depends(get_manager)):
    await manager.stop()
```
with:
```python
@router.delete("/deployments/{dep_id}", status_code=status.HTTP_204_NO_CONTENT)
async def stop_deployment(
    dep_id: int,
    manager: LifecycleManager = Depends(get_manager),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if dep_store.get_by_id(conn, dep_id) is None:
        raise HTTPException(404, f"no deployment with id {dep_id}")
    await manager.stop(dep_id)


@router.delete("/deployments", status_code=status.HTTP_204_NO_CONTENT)
async def stop_all_deployments(manager: LifecycleManager = Depends(get_manager)):
    await manager.stop()  # stops all
```

- [ ] **Step 4: Tests**

Append to `tests/unit/test_admin_endpoints.py`:

```python
@pytest.mark.asyncio
async def test_pin_unpin_deployment(app):
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
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/admin/deployments/999/pin")
    assert r.status_code == 404
```

- [ ] **Step 5: Run tests, all pass.**

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/daemon/admin.py tests/unit/test_admin_endpoints.py
git commit -m "feat(daemon): pin/unpin endpoints + delete-by-id + create flags"
```

---

## Task 10: CLI pin/unpin commands + ps shows new columns

**Files:**
- Create: `src/serve_engine/cli/pin_cmd.py`
- Create: `src/serve_engine/cli/unpin_cmd.py`
- Modify: `src/serve_engine/cli/ps_cmd.py`
- Modify: `src/serve_engine/cli/run_cmd.py`
- Modify: `src/serve_engine/cli/stop_cmd.py`
- Modify: `src/serve_engine/cli/__init__.py`

- [ ] **Step 1: Implement `pin_cmd.py`**

```python
from __future__ import annotations
import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("pin")
def pin(model_name: str = typer.Argument(...)):
    """Mark the deployment for <model_name> as pinned (never auto-evicted)."""
    deps = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/deployments"))
    matches = [d for d in deps if d.get("model_id")]  # we'll filter by model below
    # We need model_id → name mapping; fetch models
    models = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/models"))
    model = next((m for m in models if m["name"] == model_name), None)
    if model is None:
        typer.echo(f"model {model_name!r} not registered", err=True)
        raise typer.Exit(1)
    ready = [d for d in deps if d.get("status") == "ready" and d.get("model_id") == model["id"]]
    if not ready:
        typer.echo(f"no ready deployment for {model_name!r}", err=True)
        raise typer.Exit(1)
    dep_id = ready[0]["id"]
    asyncio.run(ipc.post(config.SOCK_PATH, f"/admin/deployments/{dep_id}/pin"))
    typer.echo(f"pinned deployment #{dep_id} ({model_name})")
```

- [ ] **Step 2: `unpin_cmd.py`** — same shape, calls `/unpin` endpoint instead, message says "unpinned".

```python
from __future__ import annotations
import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("unpin")
def unpin(model_name: str = typer.Argument(...)):
    """Mark the deployment for <model_name> as auto (LRU-evictable)."""
    deps = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/deployments"))
    models = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/models"))
    model = next((m for m in models if m["name"] == model_name), None)
    if model is None:
        typer.echo(f"model {model_name!r} not registered", err=True)
        raise typer.Exit(1)
    ready = [d for d in deps if d.get("status") == "ready" and d.get("model_id") == model["id"]]
    if not ready:
        typer.echo(f"no ready deployment for {model_name!r}", err=True)
        raise typer.Exit(1)
    dep_id = ready[0]["id"]
    asyncio.run(ipc.post(config.SOCK_PATH, f"/admin/deployments/{dep_id}/unpin"))
    typer.echo(f"unpinned deployment #{dep_id} ({model_name})")
```

- [ ] **Step 3: Update `ps_cmd.py` to show pinned + VRAM**

```python
from __future__ import annotations
import asyncio
import json

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("ps")
def ps(json_out: bool = typer.Option(False, "--json")):
    """List deployments and their status."""
    deps = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/deployments"))
    if json_out:
        typer.echo(json.dumps(deps, indent=2))
        return
    if not deps:
        typer.echo("no deployments")
        return
    typer.echo(
        f"{'ID':<4} {'STATUS':<10} {'PIN':<4} {'BACKEND':<8} {'GPUs':<10} "
        f"{'VRAM(MB)':<10} {'CONTAINER':<30}"
    )
    for d in deps:
        pin = "*" if d.get("pinned") else "-"
        typer.echo(
            f"{d['id']:<4} {d['status']:<10} {pin:<4} {d['backend']:<8} "
            f"{','.join(str(g) for g in d['gpu_ids']):<10} "
            f"{d.get('vram_reserved_mb', 0):<10} "
            f"{d.get('container_name') or '-':<30}"
        )
```

- [ ] **Step 4: `run_cmd.py` accepts `--pin` and `--idle-timeout`**

Insert two new options:
```python
    pin: bool = typer.Option(False, "--pin", help="Make this deployment unevictable"),
    idle_timeout_s: int = typer.Option(None, "--idle-timeout", help="Seconds idle before auto-eviction"),
```
and pass them through in the request body:
```python
    body = {
        "model_name": local_name,
        "hf_repo": hf_repo,
        "gpu_ids": gpu_ids,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "pinned": pin,
    }
    if image_tag is not None:
        body["image_tag"] = image_tag
    if idle_timeout_s is not None:
        body["idle_timeout_s"] = idle_timeout_s
```

- [ ] **Step 5: `stop_cmd.py` accepts an optional deployment id**

```python
@app.command("stop")
def stop(
    dep_id: int = typer.Argument(None, help="Deployment id (default: stop all)"),
):
    """Stop a deployment by id, or all if no id is given."""
    if dep_id is None:
        asyncio.run(ipc.delete(config.SOCK_PATH, "/admin/deployments"))
        typer.echo("all deployments stopped")
    else:
        asyncio.run(ipc.delete(config.SOCK_PATH, f"/admin/deployments/{dep_id}"))
        typer.echo(f"stopped deployment #{dep_id}")
```

- [ ] **Step 6: Register the new CLI commands**

In `src/serve_engine/cli/__init__.py`, add `pin_cmd` and `unpin_cmd` to the import block:
```python
from serve_engine.cli import (  # noqa: F401,E402
    daemon_cmd,
    logs_cmd,
    ls_cmd,
    pin_cmd,
    ps_cmd,
    pull_cmd,
    run_cmd,
    stop_cmd,
    unpin_cmd,
)
```

- [ ] **Step 7: Smoke check (no unit test for CLI shape)**

```bash
python -c "from serve_engine.cli import pin_cmd, unpin_cmd; print('import ok')"
pytest -v
ruff check src/ tests/
```
Expected: all pass, ruff clean.

- [ ] **Step 8: Commit**

```bash
git add src/serve_engine/cli/pin_cmd.py src/serve_engine/cli/unpin_cmd.py src/serve_engine/cli/ps_cmd.py src/serve_engine/cli/run_cmd.py src/serve_engine/cli/stop_cmd.py src/serve_engine/cli/__init__.py
git commit -m "feat(cli): pin/unpin commands + ps shows pinned/VRAM + stop by id"
```

---

## Task 11: Smoke script v2 (multi-model + pin/auto-swap)

**Files:**
- Modify: `scripts/smoke_e2e.sh`

The v2 script:
1. Starts the daemon.
2. Pulls `Qwen/Qwen2.5-0.5B-Instruct` as `qwen-0_5b`.
3. Runs it with `--pin`.
4. Pulls `Qwen/Qwen2.5-1.5B-Instruct` as `qwen-1_5b`.
5. Runs it with `--idle-timeout 60`.
6. Calls `/v1/chat/completions` against both models (different `model` field) — both should succeed because the proxy now routes by model name.
7. Verifies via `serve ps` that both deployments are `ready`.

- [ ] **Step 1: Replace `scripts/smoke_e2e.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Plan 02 smoke test: multi-model serving with pin + auto.
#
# Prerequisites (same as Plan 01):
#   - Docker daemon + nvidia-container-toolkit + CUDA GPUs (>=1)
#   - HF cache primed or HF_TOKEN set
#   - serve installed via `uv pip install -e ".[dev]"`

cleanup() {
    serve stop 2>/dev/null || true
    serve daemon stop 2>/dev/null || true
}
trap cleanup EXIT

serve daemon start

serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b
serve pull Qwen/Qwen2.5-1.5B-Instruct --name qwen-1_5b

# Pinned: never auto-evicted
serve run qwen-0_5b --gpu 0 --ctx 4096 --pin

# Auto: idle-evicted after 60 s with no traffic
serve run qwen-1_5b --gpu 0 --ctx 4096 --idle-timeout 60

serve ps

# Hit both models
for m in qwen-0_5b qwen-1_5b; do
    echo "--- $m ---"
    curl -sS -N "http://127.0.0.1:11500/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply: OK\"}],\"max_tokens\":4,\"stream\":false}" \
      | tee "/tmp/serve_smoke_$m.out"
    grep -q "OK" "/tmp/serve_smoke_$m.out" || { echo "FAIL: no OK from $m"; exit 1; }
done

echo "PASS"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/smoke_e2e.sh
git commit -m "test: Plan 02 smoke (pin + auto-swap + multi-model routing)"
```

---

## Verification (end of Plan 02)

After all tasks:

1. `pytest -v` — all unit + integration tests pass with no GPU.
2. `ruff check src/ tests/` — clean.
3. On real hardware (8 H100 / 1 H100): `bash scripts/smoke_e2e.sh` exits 0 and prints `PASS`.

---

## Self-review

- **Spec coverage:** Pin/auto-swap (T1, T5, T9, T10), idle eviction (T8), GPU topology (T3, T6), placement (T4, T5), KV-aware reservation (T2, T5), routing by model name (T7), container address persisted (T1, T7), public docker accessor (T5).
- **No autotune, no SGLang, no multi-tenancy, no UI** — correctly deferred to later plans.
- **Placeholder scan:** none — every code step has full code.
- **Type consistency:** `Deployment.gpu_ids: list[int]`, `Deployment.pinned: bool`, `ContainerHandle.address: str`, `Topology.gpus: list[GPUInfo]`, `PlacementRequest.tensor_parallel: int`, `Fit.gpu_ids: list[int]` — names used identically across tasks.
- **Forward-compat:** `target_concurrency` and `idle_timeout_s` on `DeploymentPlan` give Plan 03 (autotune) and Plan 06 (observability) somewhere to land.
