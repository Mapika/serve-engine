# Serving Engine - Plan 01: Walking Skeleton

**Goal:** Build a single-model serving daemon: user can `serve daemon start`, `serve pull meta-llama/Llama-3.2-1B-Instruct`, `serve run meta-llama/Llama-3.2-1B-Instruct`, and then hit `POST /v1/chat/completions` and receive a streamed response from a vLLM container that the daemon spawned and supervises.

**Architecture:** Python daemon (FastAPI + uvicorn) that exposes a public TCP port for inference requests and a Unix domain socket for CLI commands. The daemon supervises exactly one vLLM container at a time, talking to it over Docker's bridge network. Persistent state in SQLite. CLI (Typer) is a thin HTTP-over-UDS client. No auth, no multi-model, no autotune in this plan - those come in Plans 02-05.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx (async, UDS-capable), Typer, docker-py, huggingface_hub, structlog, pytest, pytest-asyncio. `uv` for packaging. Docker 24+ with `nvidia-container-toolkit` on the host. Upstream `vllm/vllm-openai` image.

**Scope explicitly NOT in this plan:** SGLang, pin/auto-swap, GPU topology / placement, autotune, API keys, fair queueing, web UI, metrics aggregation, `serve doctor`, installer script. All of these are planned in subsequent plans.

---

## File structure produced by this plan

```
serving-engine/
|-- pyproject.toml
|-- README.md
|-- src/serve_engine/
|   |-- __init__.py
|   |-- config.py                       # paths (~/.serve), constants
|   |-- store/
|   |   |-- __init__.py
|   |   |-- db.py                       # sqlite3 connection, migration runner
|   |   |-- migrations/001_initial.sql
|   |   |-- models.py                   # model registry queries
|   |   +-- deployments.py              # deployment row queries
|   |-- lifecycle/
|   |   |-- __init__.py
|   |   |-- plan.py                     # DeploymentPlan dataclass
|   |   |-- docker_client.py            # thin wrapper around docker-py
|   |   |-- downloader.py               # HF snapshot download wrapper
|   |   +-- manager.py                  # single-deployment lifecycle
|   |-- backends/
|   |   |-- __init__.py
|   |   |-- base.py                     # Backend Protocol
|   |   +-- vllm.py                     # vLLM Backend impl
|   |-- daemon/
|   |   |-- __init__.py
|   |   |-- app.py                      # FastAPI app factory
|   |   |-- admin.py                    # /admin/* routes
|   |   |-- openai_proxy.py             # /v1/* routes
|   |   +-- __main__.py                 # python -m serve_engine.daemon
|   +-- cli/
|       |-- __init__.py                 # Typer app, command registry
|       |-- ipc.py                      # httpx client for UDS
|       |-- daemon_cmd.py               # daemon start/stop/status
|       |-- pull_cmd.py
|       |-- run_cmd.py
|       |-- stop_cmd.py
|       |-- ps_cmd.py
|       |-- ls_cmd.py
|       +-- logs_cmd.py
+-- tests/
    |-- conftest.py
    |-- unit/
    |   |-- test_store.py
    |   |-- test_docker_client.py
    |   |-- test_vllm_backend.py
    |   |-- test_lifecycle_manager.py
    |   +-- test_admin_endpoints.py
    +-- integration/
        +-- test_openai_proxy.py        # mock upstream engine
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/serve_engine/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_smoke.py`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "serve-engine"
version = "0.0.1"
description = "Single-node multi-user inference orchestrator over vLLM and SGLang"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "httpx>=0.27",
    "typer>=0.12",
    "docker>=7.1",
    "huggingface_hub>=0.24",
    "pydantic>=2.7",
    "structlog>=24.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "anyio>=4.4",
]

[project.scripts]
serve = "serve_engine.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/serve_engine"]

[tool.hatch.build.targets.wheel.force-include]
"src/serve_engine/store/migrations" = "serve_engine/store/migrations"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
exclude = ["tests/"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write `src/serve_engine/__init__.py`**

```python
__version__ = "0.0.1"
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
import asyncio
import pytest


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

- [ ] **Step 4: Write `tests/unit/test_smoke.py`**

```python
def test_package_imports():
    import serve_engine
    assert serve_engine.__version__ == "0.0.1"
```

- [ ] **Step 5: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
*.egg-info/
.serve/
```

- [ ] **Step 6: Write `README.md`** (placeholder; final README written in Task 19)

```markdown
# serve-engine

Single-node multi-user inference orchestrator over vLLM (and later SGLang).

Work in progress - see `docs/design/specs/` for the design and
`docs/design/plans/` for implementation plans.
```

- [ ] **Step 7: Verify the project installs and the smoke test passes**

Run:
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/unit/test_smoke.py -v
ruff check src/ tests/
```
Expected: smoke test PASSES; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/serve_engine/__init__.py tests/ README.md .gitignore
git commit -m "feat(scaffold): initial project structure"
```

---

## Task 2: SQLite store + schema

**Files:**
- Create: `src/serve_engine/config.py`
- Create: `src/serve_engine/store/__init__.py`
- Create: `src/serve_engine/store/db.py`
- Create: `src/serve_engine/store/migrations/001_initial.sql`
- Create: `tests/unit/test_store.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_store.py`:
```python
import sqlite3
import pytest

from serve_engine.store import db


def test_init_schema_creates_tables(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.init_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r[0] for r in rows}
    assert "models" in table_names
    assert "deployments" in table_names
    assert "_migrations" in table_names


def test_init_schema_is_idempotent(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.init_schema(conn)
    db.init_schema(conn)  # second call must not error

    applied = conn.execute(
        "SELECT COUNT(*) FROM _migrations WHERE filename='001_initial.sql'"
    ).fetchone()[0]
    assert applied == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_store.py -v`
Expected: FAIL - module `serve_engine.store.db` does not exist.

- [ ] **Step 3: Write `src/serve_engine/config.py`**

```python
from __future__ import annotations
import os
from pathlib import Path

SERVE_DIR = Path(os.environ.get("SERVE_HOME", Path.home() / ".serve"))
MODELS_DIR = SERVE_DIR / "models"
LOGS_DIR = SERVE_DIR / "logs"
DB_PATH = SERVE_DIR / "db.sqlite"
SOCK_PATH = SERVE_DIR / "sock"

DEFAULT_PUBLIC_HOST = "127.0.0.1"
DEFAULT_PUBLIC_PORT = 11500

DOCKER_NETWORK_NAME = "serve-engines"
```

- [ ] **Step 4: Write `src/serve_engine/store/__init__.py`**

```python
```
(empty)

- [ ] **Step 5: Write `src/serve_engine/store/migrations/001_initial.sql`**

```sql
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    hf_repo TEXT NOT NULL,
    revision TEXT NOT NULL DEFAULT 'main',
    local_path TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    backend TEXT NOT NULL,
    image_tag TEXT NOT NULL,
    gpu_ids TEXT NOT NULL DEFAULT '',
    tensor_parallel INTEGER NOT NULL DEFAULT 1,
    max_model_len INTEGER,
    dtype TEXT NOT NULL DEFAULT 'auto',
    container_id TEXT,
    container_name TEXT,
    container_port INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    last_error TEXT,
    started_at TIMESTAMP,
    last_request_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
```

- [ ] **Step 6: Write `src/serve_engine/store/db.py`**

```python
from __future__ import annotations
import sqlite3
from importlib.resources import files
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def init_schema(conn: sqlite3.Connection) -> None:
    _ensure_migrations_table(conn)
    mig_dir = files("serve_engine.store.migrations")
    for entry in sorted(mig_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".sql"):
            continue
        already = conn.execute(
            "SELECT 1 FROM _migrations WHERE filename=?", (entry.name,)
        ).fetchone()
        if already:
            continue
        sql = entry.read_text()
        conn.executescript(sql)
        conn.execute("INSERT INTO _migrations (filename) VALUES (?)", (entry.name,))
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/unit/test_store.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/serve_engine/config.py src/serve_engine/store/ tests/unit/test_store.py
git commit -m "feat(store): SQLite connection + initial schema with migration runner"
```

---

## Task 3: Model registry queries

**Files:**
- Create: `src/serve_engine/store/models.py`
- Modify: `tests/unit/test_store.py` (add tests)

- [ ] **Step 1: Write the failing test (append to `tests/unit/test_store.py`)**

```python
from serve_engine.store import models as model_store


def _fresh(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.init_schema(conn)
    return conn


def test_add_and_get_model(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    assert m.id is not None
    assert m.name == "llama-1b"
    assert m.revision == "main"

    fetched = model_store.get_by_name(conn, "llama-1b")
    assert fetched is not None
    assert fetched.id == m.id


def test_add_duplicate_model_raises(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="x", hf_repo="org/x")
    with pytest.raises(model_store.AlreadyExists):
        model_store.add(conn, name="x", hf_repo="org/x")


def test_list_models_empty(tmp_path):
    conn = _fresh(tmp_path)
    assert model_store.list_all(conn) == []


def test_list_models_returns_in_creation_order(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="a", hf_repo="org/a")
    model_store.add(conn, name="b", hf_repo="org/b")
    rows = model_store.list_all(conn)
    assert [m.name for m in rows] == ["a", "b"]


def test_set_local_path(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    model_store.set_local_path(conn, m.id, "/var/x")
    fetched = model_store.get_by_name(conn, "x")
    assert fetched.local_path == "/var/x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_store.py -v`
Expected: FAIL on the import of `models`.

- [ ] **Step 3: Implement `src/serve_engine/store/models.py`**

```python
from __future__ import annotations
import sqlite3
from dataclasses import dataclass


class AlreadyExists(Exception):
    pass


@dataclass(frozen=True)
class Model:
    id: int
    name: str
    hf_repo: str
    revision: str
    local_path: str | None


def _row_to_model(row: sqlite3.Row) -> Model:
    return Model(
        id=row["id"],
        name=row["name"],
        hf_repo=row["hf_repo"],
        revision=row["revision"],
        local_path=row["local_path"],
    )


def add(
    conn: sqlite3.Connection,
    *,
    name: str,
    hf_repo: str,
    revision: str = "main",
) -> Model:
    try:
        cur = conn.execute(
            "INSERT INTO models (name, hf_repo, revision) VALUES (?, ?, ?)",
            (name, hf_repo, revision),
        )
    except sqlite3.IntegrityError as e:
        raise AlreadyExists(f"model {name!r} already exists") from e
    return Model(id=cur.lastrowid, name=name, hf_repo=hf_repo, revision=revision, local_path=None)


def get_by_name(conn: sqlite3.Connection, name: str) -> Model | None:
    row = conn.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
    return _row_to_model(row) if row else None


def get_by_id(conn: sqlite3.Connection, model_id: int) -> Model | None:
    row = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    return _row_to_model(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Model]:
    rows = conn.execute("SELECT * FROM models ORDER BY id").fetchall()
    return [_row_to_model(r) for r in rows]


def set_local_path(conn: sqlite3.Connection, model_id: int, path: str) -> None:
    conn.execute("UPDATE models SET local_path=? WHERE id=?", (path, model_id))


def delete(conn: sqlite3.Connection, model_id: int) -> None:
    conn.execute("DELETE FROM models WHERE id=?", (model_id,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/store/models.py tests/unit/test_store.py
git commit -m "feat(store): model registry CRUD"
```

---

## Task 4: Deployment row queries

**Files:**
- Create: `src/serve_engine/store/deployments.py`
- Modify: `tests/unit/test_store.py` (add tests)

- [ ] **Step 1: Write the failing test (append to `tests/unit/test_store.py`)**

```python
from serve_engine.store import deployments as dep_store


def test_create_deployment(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    d = dep_store.create(
        conn,
        model_id=m.id,
        backend="vllm",
        image_tag="vllm/vllm-openai:v0.7.3",
        gpu_ids=[0],
        tensor_parallel=1,
        max_model_len=8192,
        dtype="bf16",
    )
    assert d.id is not None
    assert d.status == "pending"
    assert d.gpu_ids == [0]


def test_update_deployment_status(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=8192, dtype="auto",
    )
    dep_store.update_status(conn, d.id, "loading")
    refreshed = dep_store.get_by_id(conn, d.id)
    assert refreshed.status == "loading"


def test_set_container_info(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=8192, dtype="auto",
    )
    dep_store.set_container(conn, d.id, container_id="abc", container_name="vllm-x", container_port=8000)
    refreshed = dep_store.get_by_id(conn, d.id)
    assert refreshed.container_id == "abc"
    assert refreshed.container_port == 8000


def test_find_active(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    assert dep_store.find_active(conn) is None
    d = dep_store.create(
        conn, model_id=m.id, backend="vllm", image_tag="img:v1",
        gpu_ids=[0], tensor_parallel=1, max_model_len=8192, dtype="auto",
    )
    dep_store.update_status(conn, d.id, "ready")
    found = dep_store.find_active(conn)
    assert found is not None and found.id == d.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_store.py -v`
Expected: FAIL - `deployments` module missing.

- [ ] **Step 3: Implement `src/serve_engine/store/deployments.py`**

```python
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import Literal

Status = Literal["pending", "loading", "ready", "stopping", "stopped", "failed"]
ACTIVE_STATUSES: tuple[Status, ...] = ("pending", "loading", "ready")


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
    status: Status
    last_error: str | None


def _row_to_dep(row: sqlite3.Row) -> Deployment:
    gpu_csv = row["gpu_ids"] or ""
    gpu_ids = [int(x) for x in gpu_csv.split(",") if x]
    return Deployment(
        id=row["id"],
        model_id=row["model_id"],
        backend=row["backend"],
        image_tag=row["image_tag"],
        gpu_ids=gpu_ids,
        tensor_parallel=row["tensor_parallel"],
        max_model_len=row["max_model_len"],
        dtype=row["dtype"],
        container_id=row["container_id"],
        container_name=row["container_name"],
        container_port=row["container_port"],
        status=row["status"],
        last_error=row["last_error"],
    )


def create(
    conn: sqlite3.Connection,
    *,
    model_id: int,
    backend: str,
    image_tag: str,
    gpu_ids: list[int],
    tensor_parallel: int,
    max_model_len: int | None,
    dtype: str,
) -> Deployment:
    gpu_csv = ",".join(str(g) for g in gpu_ids)
    cur = conn.execute(
        """
        INSERT INTO deployments
            (model_id, backend, image_tag, gpu_ids, tensor_parallel, max_model_len, dtype)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (model_id, backend, image_tag, gpu_csv, tensor_parallel, max_model_len, dtype),
    )
    return get_by_id(conn, cur.lastrowid)  # type: ignore[return-value]


def get_by_id(conn: sqlite3.Connection, dep_id: int) -> Deployment | None:
    row = conn.execute("SELECT * FROM deployments WHERE id=?", (dep_id,)).fetchone()
    return _row_to_dep(row) if row else None


def update_status(
    conn: sqlite3.Connection,
    dep_id: int,
    status: Status,
    *,
    last_error: str | None = None,
) -> None:
    if last_error is not None:
        conn.execute(
            "UPDATE deployments SET status=?, last_error=? WHERE id=?",
            (status, last_error, dep_id),
        )
    else:
        conn.execute("UPDATE deployments SET status=? WHERE id=?", (status, dep_id))


def set_container(
    conn: sqlite3.Connection,
    dep_id: int,
    *,
    container_id: str,
    container_name: str,
    container_port: int,
) -> None:
    conn.execute(
        """
        UPDATE deployments
        SET container_id=?, container_name=?, container_port=?, started_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (container_id, container_name, container_port, dep_id),
    )


def find_active(conn: sqlite3.Connection) -> Deployment | None:
    placeholders = ",".join(["?"] * len(ACTIVE_STATUSES))
    row = conn.execute(
        f"SELECT * FROM deployments WHERE status IN ({placeholders}) ORDER BY id DESC LIMIT 1",
        ACTIVE_STATUSES,
    ).fetchone()
    return _row_to_dep(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Deployment]:
    rows = conn.execute("SELECT * FROM deployments ORDER BY id").fetchall()
    return [_row_to_dep(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/store/deployments.py tests/unit/test_store.py
git commit -m "feat(store): deployment row CRUD"
```

---

## Task 5: DeploymentPlan dataclass

**Files:**
- Create: `src/serve_engine/lifecycle/__init__.py`
- Create: `src/serve_engine/lifecycle/plan.py`
- Create: `tests/unit/test_plan.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_plan.py`:
```python
import pytest
from serve_engine.lifecycle.plan import DeploymentPlan


def test_plan_basic_fields():
    p = DeploymentPlan(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="vllm",
        image_tag="vllm/vllm-openai:v0.7.3",
        gpu_ids=[0],
        max_model_len=8192,
    )
    assert p.tensor_parallel == 1
    assert p.dtype == "auto"
    assert p.gpu_memory_utilization == 0.9


def test_plan_tensor_parallel_must_match_gpu_count():
    with pytest.raises(ValueError, match="tensor_parallel"):
        DeploymentPlan(
            model_name="x",
            hf_repo="org/x",
            revision="main",
            backend="vllm",
            image_tag="vllm/vllm-openai:v0.7.3",
            gpu_ids=[0, 1],
            tensor_parallel=4,
            max_model_len=8192,
        )


def test_plan_tensor_parallel_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        DeploymentPlan(
            model_name="x",
            hf_repo="org/x",
            revision="main",
            backend="vllm",
            image_tag="img:v1",
            gpu_ids=[0, 1, 2],
            tensor_parallel=3,
            max_model_len=8192,
        )


def test_plan_backend_must_be_supported():
    with pytest.raises(ValueError, match="backend"):
        DeploymentPlan(
            model_name="x",
            hf_repo="org/x",
            revision="main",
            backend="trt-llm",  # not in plan 01
            image_tag="img:v1",
            gpu_ids=[0],
            max_model_len=8192,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_plan.py -v`
Expected: FAIL - `DeploymentPlan` missing.

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/__init__.py`** (empty file)

```python
```

- [ ] **Step 4: Implement `src/serve_engine/lifecycle/plan.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

SUPPORTED_BACKENDS = ("vllm",)  # Plan 04 adds "sglang"
SUPPORTED_DTYPES = ("auto", "bf16", "fp16", "fp8")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class DeploymentPlan:
    model_name: str
    hf_repo: str
    revision: str
    backend: Literal["vllm"]
    image_tag: str
    gpu_ids: list[int]
    max_model_len: int
    tensor_parallel: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    extra_args: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend {self.backend!r} not supported in Plan 01 "
                f"(supported: {SUPPORTED_BACKENDS})"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype {self.dtype!r} not in {SUPPORTED_DTYPES}")
        if self.tensor_parallel < 1:
            raise ValueError("tensor_parallel must be >= 1")
        if not _is_power_of_two(self.tensor_parallel):
            raise ValueError("tensor_parallel must be a power of 2")
        if self.tensor_parallel != len(self.gpu_ids):
            raise ValueError(
                "tensor_parallel must equal len(gpu_ids); "
                f"got TP={self.tensor_parallel}, gpus={self.gpu_ids}"
            )
        if not 0.1 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in [0.1, 1.0]")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_plan.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/lifecycle/ tests/unit/test_plan.py
git commit -m "feat(lifecycle): DeploymentPlan dataclass with validation"
```

---

## Task 6: vLLM backend

**Files:**
- Create: `src/serve_engine/backends/__init__.py`
- Create: `src/serve_engine/backends/base.py`
- Create: `src/serve_engine/backends/vllm.py`
- Create: `tests/unit/test_vllm_backend.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vllm_backend.py`:
```python
from serve_engine.backends.vllm import VLLMBackend
from serve_engine.lifecycle.plan import DeploymentPlan


def _plan(**overrides):
    base = dict(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="vllm",
        image_tag="vllm/vllm-openai:v0.7.3",
        gpu_ids=[0],
        max_model_len=8192,
    )
    base.update(overrides)
    return DeploymentPlan(**base)


def test_build_argv_minimum():
    argv = VLLMBackend().build_argv(_plan(), local_model_path="/models/llama-1b")
    assert argv[0] == "--model"
    assert argv[1] == "/models/llama-1b"
    assert "--tensor-parallel-size" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert "--max-model-len" in argv
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert "--enable-prefix-caching" in argv
    assert "--enable-chunked-prefill" in argv
    assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8000"


def test_build_argv_tp_4():
    argv = VLLMBackend().build_argv(
        _plan(gpu_ids=[0, 1, 2, 3], tensor_parallel=4),
        local_model_path="/models/x",
    )
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"


def test_container_kwargs_gpu_request():
    kw = VLLMBackend().container_kwargs(_plan(gpu_ids=[2, 3], tensor_parallel=2))
    assert kw["device_requests"][0]["device_ids"] == ["2", "3"]
    assert kw["ipc_mode"] == "host"
    assert kw["shm_size"] == "2g"
    assert kw["ulimits"][0].name == "memlock"


def test_default_image():
    assert VLLMBackend.image_default.startswith("vllm/vllm-openai:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vllm_backend.py -v`
Expected: FAIL - module missing.

- [ ] **Step 3: Implement `src/serve_engine/backends/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Implement `src/serve_engine/backends/base.py`**

```python
from __future__ import annotations
from typing import Protocol

from serve_engine.lifecycle.plan import DeploymentPlan


class Backend(Protocol):
    name: str
    image_default: str
    health_path: str
    openai_base: str
    metrics_path: str

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]: ...
    def container_env(self, plan: DeploymentPlan) -> dict[str, str]: ...
    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]: ...
```

- [ ] **Step 5: Implement `src/serve_engine/backends/vllm.py`**

```python
from __future__ import annotations
from typing import ClassVar

from docker.types import Ulimit  # type: ignore[import-untyped]

from serve_engine.lifecycle.plan import DeploymentPlan

ENGINE_INTERNAL_PORT = 8000


class VLLMBackend:
    name: ClassVar[str] = "vllm"
    image_default: ClassVar[str] = "vllm/vllm-openai:v0.7.3"
    health_path: ClassVar[str] = "/health"
    openai_base: ClassVar[str] = "/v1"
    metrics_path: ClassVar[str] = "/metrics"

    def build_argv(self, plan: DeploymentPlan, *, local_model_path: str) -> list[str]:
        argv: list[str] = [
            "--model", local_model_path,
            "--tensor-parallel-size", str(plan.tensor_parallel),
            "--max-model-len", str(plan.max_model_len),
            "--gpu-memory-utilization", str(plan.gpu_memory_utilization),
            "--dtype", plan.dtype,
            "--host", "0.0.0.0",
            "--port", str(ENGINE_INTERNAL_PORT),
            "--served-model-name", plan.model_name,
        ]
        if plan.enable_prefix_caching:
            argv.append("--enable-prefix-caching")
        if plan.enable_chunked_prefill:
            argv.append("--enable-chunked-prefill")
        for k, v in plan.extra_args.items():
            argv.extend([k, v])
        return argv

    def container_env(self, plan: DeploymentPlan) -> dict[str, str]:
        # Engines read HF token if a gated model is mounted by reference, but our
        # primary path mounts pre-downloaded weights so this is usually unused.
        return {}

    def container_kwargs(self, plan: DeploymentPlan) -> dict[str, object]:
        return {
            "device_requests": [
                {
                    "Driver": "nvidia",
                    "device_ids": [str(g) for g in plan.gpu_ids],
                    "Capabilities": [["gpu"]],
                }
            ],
            "ipc_mode": "host",
            "shm_size": "2g",
            "ulimits": [Ulimit(name="memlock", soft=-1, hard=-1)],
        }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_vllm_backend.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/serve_engine/backends/ tests/unit/test_vllm_backend.py
git commit -m "feat(backends): vLLM backend (argv + container kwargs)"
```

---

## Task 7: Docker client wrapper

**Files:**
- Create: `src/serve_engine/lifecycle/docker_client.py`
- Create: `tests/unit/test_docker_client.py`

The wrapper is a thin facade over docker-py so the lifecycle manager can be unit-tested with a mock and the real docker-py noise is contained.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_docker_client.py`:
```python
from unittest.mock import MagicMock

import pytest

from serve_engine.lifecycle.docker_client import DockerClient, ContainerHandle


@pytest.fixture
def fake_docker():
    """A MagicMock that imitates the bits of docker.from_env() we touch."""
    client = MagicMock()
    container = MagicMock()
    container.id = "abc123"
    container.name = "vllm-llama-1b"
    container.attrs = {"NetworkSettings": {"Networks": {"serve-engines": {"IPAddress": "172.20.0.5"}}}}
    client.containers.run.return_value = container
    return client


def test_run_container_returns_handle(fake_docker):
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    handle = dc.run(
        image="vllm/vllm-openai:v0.7.3",
        name="vllm-llama-1b",
        command=["--model", "/models/x"],
        environment={},
        kwargs={"ipc_mode": "host"},
        volumes={"/host/models": {"bind": "/models", "mode": "ro"}},
        internal_port=8000,
    )
    assert isinstance(handle, ContainerHandle)
    assert handle.id == "abc123"
    assert handle.address == "vllm-llama-1b"  # container-name addressing on the bridge
    assert handle.port == 8000


def test_run_creates_network_if_missing(fake_docker):
    fake_docker.networks.get.side_effect = Exception("not found")
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.ensure_network()
    fake_docker.networks.create.assert_called_once_with("serve-engines", driver="bridge")


def test_run_skips_network_create_if_present(fake_docker):
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.ensure_network()
    fake_docker.networks.create.assert_not_called()


def test_stop_calls_remove(fake_docker):
    container = MagicMock()
    fake_docker.containers.get.return_value = container
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.stop("abc123", timeout=10)
    container.stop.assert_called_once_with(timeout=10)
    container.remove.assert_called_once()


def test_stop_is_idempotent_for_missing_container(fake_docker):
    from docker.errors import NotFound
    fake_docker.containers.get.side_effect = NotFound("gone")
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.stop("abc123", timeout=10)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_docker_client.py -v`
Expected: FAIL - module missing.

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/docker_client.py`**

```python
from __future__ import annotations
import logging
from dataclasses import dataclass

import docker  # type: ignore[import-untyped]
from docker.errors import NotFound  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContainerHandle:
    id: str
    name: str
    address: str       # host or container name reachable from daemon
    port: int          # port on `address` to talk HTTP to


class DockerClient:
    def __init__(self, *, client: object | None = None, network_name: str):
        self._client = client or docker.from_env()
        self._network_name = network_name

    def ensure_network(self) -> None:
        try:
            self._client.networks.get(self._network_name)
        except Exception:
            log.info("creating docker network %s", self._network_name)
            self._client.networks.create(self._network_name, driver="bridge")

    def run(
        self,
        *,
        image: str,
        name: str,
        command: list[str],
        environment: dict[str, str],
        kwargs: dict[str, object],
        volumes: dict[str, dict[str, str]],
        internal_port: int,
    ) -> ContainerHandle:
        container = self._client.containers.run(
            image=image,
            command=command,
            name=name,
            environment=environment,
            volumes=volumes,
            network=self._network_name,
            detach=True,
            **kwargs,
        )
        return ContainerHandle(
            id=container.id,
            name=name,
            address=name,  # talk by container name on the bridge
            port=internal_port,
        )

    def stop(self, container_id: str, *, timeout: int = 30) -> None:
        try:
            c = self._client.containers.get(container_id)
        except NotFound:
            return
        c.stop(timeout=timeout)
        c.remove()

    def stream_logs(self, container_id: str, *, follow: bool = False):
        c = self._client.containers.get(container_id)
        return c.logs(stream=True, follow=follow)

    def pull(self, image: str) -> None:
        self._client.images.pull(image)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_docker_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/lifecycle/docker_client.py tests/unit/test_docker_client.py
git commit -m "feat(lifecycle): Docker client wrapper for container lifecycle"
```

---

## Task 8: HF downloader wrapper

**Files:**
- Create: `src/serve_engine/lifecycle/downloader.py`
- Create: `tests/unit/test_downloader.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_downloader.py`:
```python
from unittest.mock import patch

from serve_engine.lifecycle.downloader import download_model


def test_download_model_calls_snapshot_download(tmp_path):
    with patch("serve_engine.lifecycle.downloader.snapshot_download") as mock_sd:
        mock_sd.return_value = str(tmp_path / "x")
        result = download_model(
            hf_repo="meta-llama/Llama-3.2-1B-Instruct",
            revision="main",
            cache_dir=tmp_path,
        )
    assert result == str(tmp_path / "x")
    mock_sd.assert_called_once()
    kwargs = mock_sd.call_args.kwargs
    assert kwargs["repo_id"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert kwargs["revision"] == "main"
    assert kwargs["cache_dir"] == str(tmp_path)


def test_download_model_progress_callback(tmp_path):
    seen: list[str] = []
    def cb(msg: str) -> None:
        seen.append(msg)

    with patch("serve_engine.lifecycle.downloader.snapshot_download") as mock_sd:
        mock_sd.return_value = str(tmp_path / "x")
        download_model(
            hf_repo="org/x",
            revision="main",
            cache_dir=tmp_path,
            on_event=cb,
        )
    # We expect start and done events
    assert any("download started" in s for s in seen)
    assert any("download complete" in s for s in seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_downloader.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/downloader.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

ProgressFn = Callable[[str], None]


def download_model(
    *,
    hf_repo: str,
    revision: str,
    cache_dir: Path,
    on_event: ProgressFn | None = None,
) -> str:
    def emit(msg: str) -> None:
        if on_event is not None:
            on_event(msg)

    emit(f"download started: {hf_repo}@{revision}")
    path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    emit(f"download complete: {path}")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_downloader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/lifecycle/downloader.py tests/unit/test_downloader.py
git commit -m "feat(lifecycle): HF snapshot download wrapper with progress events"
```

---

## Task 9: Lifecycle manager (single deployment)

**Files:**
- Create: `src/serve_engine/lifecycle/manager.py`
- Create: `tests/unit/test_lifecycle_manager.py`

The manager owns:
1. There is at most one *active* deployment at a time (Plan 01 simplification).
2. To load a new deployment, the previous one is stopped first.
3. Health-check polling waits for the engine's `/health` to return 200 before marking ready.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_lifecycle_manager.py`:
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.backends.vllm import VLLMBackend
from serve_engine.store import db, models as model_store, deployments as dep_store


def _make_plan() -> DeploymentPlan:
    return DeploymentPlan(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="vllm",
        image_tag="vllm/vllm-openai:v0.7.3",
        gpu_ids=[0],
        max_model_len=8192,
    )


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    return c


def test_load_starts_engine_and_marks_ready(conn, monkeypatch, tmp_path):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="vllm-llama-1b", address="vllm-llama-1b", port=8000
    )

    async def fake_wait_healthy(*args, **kwargs):
        return True
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(side_effect=fake_wait_healthy)
    )

    async def fake_download(**kw):
        return str(tmp_path / "weights")
    monkeypatch.setattr("serve_engine.lifecycle.manager.download_model_async", AsyncMock(side_effect=fake_download))

    mgr = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )

    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    dep = asyncio.run(mgr.load(_make_plan()))
    assert dep.status == "ready"
    assert dep.container_id == "cid"
    docker_client.run.assert_called_once()


def test_load_stops_previous_deployment(conn, monkeypatch, tmp_path):
    docker_client = MagicMock()
    docker_client.run.side_effect = [
        ContainerHandle(id="cid1", name="x1", address="x1", port=8000),
        ContainerHandle(id="cid2", name="x2", address="x2", port=8000),
    ]
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )

    mgr = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    asyncio.run(mgr.load(_make_plan()))
    asyncio.run(mgr.load(_make_plan()))
    # The first deployment must have been stopped
    docker_client.stop.assert_called_once_with("cid1", timeout=30)


def test_load_marks_failed_on_unhealthy(conn, monkeypatch, tmp_path):
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(id="cid", name="x", address="x", port=8000)
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )
    mgr = LifecycleManager(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )
    model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    with pytest.raises(RuntimeError, match="did not become healthy"):
        asyncio.run(mgr.load(_make_plan()))
    docker_client.stop.assert_called_once()  # cleanup
    found = dep_store.find_active(conn)
    assert found is None  # failed deployment is not active
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_lifecycle_manager.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/serve_engine/lifecycle/manager.py`**

```python
from __future__ import annotations
import asyncio
import logging
import sqlite3
from pathlib import Path

import httpx

from serve_engine.backends.base import Backend
from serve_engine.lifecycle.docker_client import ContainerHandle, DockerClient
from serve_engine.lifecycle.downloader import download_model
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store

log = logging.getLogger(__name__)


async def download_model_async(**kwargs) -> str:
    # snapshot_download is blocking; offload to a thread
    return await asyncio.to_thread(download_model, **kwargs)


async def wait_healthy(url: str, *, timeout_s: float = 600.0, interval_s: float = 2.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(interval_s)
    return False


class LifecycleManager:
    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        docker_client: DockerClient,
        backends: dict[str, Backend],
        models_dir: Path,
        load_timeout_s: float = 600.0,
    ):
        self._conn = conn
        self._docker = docker_client
        self._backends = backends
        self._models_dir = models_dir
        self._load_timeout_s = load_timeout_s
        self._lock = asyncio.Lock()

    @property
    def active(self):
        return dep_store.find_active(self._conn)

    async def load(self, plan: DeploymentPlan):
        async with self._lock:
            # Ensure model record exists
            model = model_store.get_by_name(self._conn, plan.model_name)
            if model is None:
                model = model_store.add(
                    self._conn, name=plan.model_name, hf_repo=plan.hf_repo, revision=plan.revision
                )

            # Stop any current active deployment
            current = dep_store.find_active(self._conn)
            if current is not None:
                await self._stop_locked(current.id)

            # Make sure weights are present locally
            local_path = model.local_path
            if local_path is None:
                local_path = await download_model_async(
                    hf_repo=plan.hf_repo,
                    revision=plan.revision,
                    cache_dir=self._models_dir,
                )
                model_store.set_local_path(self._conn, model.id, local_path)

            # Create deployment row
            dep = dep_store.create(
                self._conn,
                model_id=model.id,
                backend=plan.backend,
                image_tag=plan.image_tag,
                gpu_ids=plan.gpu_ids,
                tensor_parallel=plan.tensor_parallel,
                max_model_len=plan.max_model_len,
                dtype=plan.dtype,
            )
            dep_store.update_status(self._conn, dep.id, "loading")

            backend = self._backends[plan.backend]
            handle = self._docker.run(
                image=plan.image_tag,
                name=f"serve-{plan.backend}-{plan.model_name}-{dep.id}",
                command=backend.build_argv(plan, local_model_path="/model"),
                environment=backend.container_env(plan),
                kwargs=backend.container_kwargs(plan),
                volumes={local_path: {"bind": "/model", "mode": "ro"}},
                internal_port=8000,
            )
            dep_store.set_container(
                self._conn,
                dep.id,
                container_id=handle.id,
                container_name=handle.name,
                container_port=handle.port,
            )

            health_url = f"http://{handle.address}:{handle.port}{backend.health_path}"
            ok = await wait_healthy(health_url, timeout_s=self._load_timeout_s)
            if not ok:
                self._docker.stop(handle.id, timeout=10)
                dep_store.update_status(
                    self._conn, dep.id, "failed",
                    last_error="engine did not become healthy within load timeout",
                )
                raise RuntimeError("engine did not become healthy within load timeout")

            dep_store.update_status(self._conn, dep.id, "ready")
            return dep_store.get_by_id(self._conn, dep.id)

    async def stop(self) -> None:
        async with self._lock:
            current = dep_store.find_active(self._conn)
            if current is None:
                return
            await self._stop_locked(current.id)

    async def _stop_locked(self, dep_id: int) -> None:
        dep = dep_store.get_by_id(self._conn, dep_id)
        if dep is None:
            return
        dep_store.update_status(self._conn, dep.id, "stopping")
        if dep.container_id:
            self._docker.stop(dep.container_id, timeout=30)
        dep_store.update_status(self._conn, dep.id, "stopped")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_lifecycle_manager.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/serve_engine/lifecycle/manager.py tests/unit/test_lifecycle_manager.py
git commit -m "feat(lifecycle): single-deployment lifecycle manager"
```

---

## Task 10: FastAPI app factory + admin routes

**Files:**
- Create: `src/serve_engine/daemon/__init__.py` (empty)
- Create: `src/serve_engine/daemon/app.py`
- Create: `src/serve_engine/daemon/admin.py`
- Create: `tests/unit/test_admin_endpoints.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_admin_endpoints.py`:
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import httpx
import pytest

from serve_engine.daemon.app import build_app
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.backends.vllm import VLLMBackend
from serve_engine.store import db


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "weights")),
    )
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="x", port=8000
    )
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_admin_endpoints.py -v`
Expected: FAIL - `build_app` missing.

- [ ] **Step 3: Implement `src/serve_engine/daemon/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Implement `src/serve_engine/daemon/admin.py`**

```python
from __future__ import annotations
import sqlite3
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from serve_engine.backends.base import Backend
from serve_engine.lifecycle.manager import LifecycleManager
from serve_engine.lifecycle.plan import DeploymentPlan
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store


router = APIRouter(prefix="/admin")


def get_manager(request: Request) -> LifecycleManager:
    return request.app.state.manager


def get_conn(request: Request) -> sqlite3.Connection:
    return request.app.state.conn


def get_backends(request: Request) -> dict[str, Backend]:
    return request.app.state.backends


class CreateDeploymentRequest(BaseModel):
    model_name: str
    hf_repo: str
    revision: str = "main"
    backend: str = "vllm"
    image_tag: str | None = None
    gpu_ids: list[int]
    tensor_parallel: int | None = None
    max_model_len: int = 8192
    dtype: str = "auto"


class CreateModelRequest(BaseModel):
    name: str
    hf_repo: str
    revision: str = "main"


@router.get("/deployments")
def list_deployments(conn: sqlite3.Connection = Depends(get_conn)):
    return [
        {**asdict(d), "gpu_ids": d.gpu_ids}
        for d in dep_store.list_all(conn)
    ]


@router.post("/deployments", status_code=status.HTTP_201_CREATED)
async def create_deployment(
    body: CreateDeploymentRequest,
    manager: LifecycleManager = Depends(get_manager),
    backends: dict[str, Backend] = Depends(get_backends),
):
    if body.backend not in backends:
        raise HTTPException(400, f"backend {body.backend!r} not supported")
    backend = backends[body.backend]
    image_tag = body.image_tag or backend.image_default
    tp = body.tensor_parallel or len(body.gpu_ids)
    try:
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
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    dep = await manager.load(plan)
    return {**asdict(dep), "gpu_ids": dep.gpu_ids}


@router.delete("/deployments/current", status_code=status.HTTP_204_NO_CONTENT)
async def stop_current(manager: LifecycleManager = Depends(get_manager)):
    await manager.stop()


@router.get("/models")
def list_models(conn: sqlite3.Connection = Depends(get_conn)):
    return [asdict(m) for m in model_store.list_all(conn)]


@router.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(
    body: CreateModelRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    try:
        m = model_store.add(conn, name=body.name, hf_repo=body.hf_repo, revision=body.revision)
    except model_store.AlreadyExists as e:
        raise HTTPException(409, str(e)) from e
    return asdict(m)


@router.delete("/models/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(name: str, conn: sqlite3.Connection = Depends(get_conn)):
    m = model_store.get_by_name(conn, name)
    if m is None:
        raise HTTPException(404, f"model {name!r} not found")
    model_store.delete(conn, m.id)
```

- [ ] **Step 5: Implement `src/serve_engine/daemon/app.py`**

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_admin_endpoints.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/serve_engine/daemon/__init__.py src/serve_engine/daemon/app.py src/serve_engine/daemon/admin.py tests/unit/test_admin_endpoints.py
git commit -m "feat(daemon): FastAPI app + /admin/* routes for models and deployments"
```

---

## Task 11: OpenAI proxy with streaming

**Files:**
- Create: `src/serve_engine/daemon/openai_proxy.py`
- Modify: `src/serve_engine/daemon/app.py` (mount router)
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_openai_proxy.py`

The proxy:
1. Receives a request on `/v1/chat/completions`.
2. Looks up the currently-active deployment via the store.
3. Forwards the request body unchanged to the engine, preserving streaming.
4. Streams the response body back to the client.

If no deployment is active or `model` in the body doesn't match the active one, return a clean error.

- [ ] **Step 1: Write the failing test**

`tests/integration/__init__.py`: (empty)

`tests/integration/test_openai_proxy.py`:
```python
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from serve_engine.daemon.app import build_app
from serve_engine.lifecycle.docker_client import ContainerHandle
from serve_engine.backends.vllm import VLLMBackend
from serve_engine.store import db


class FakeEngineApp:
    """A tiny ASGI app pretending to be the upstream engine."""

    def __init__(self, response_chunks: list[bytes], status_code: int = 200):
        self.chunks = response_chunks
        self.status_code = status_code
        self.last_request_body: bytes | None = None

    async def __call__(self, scope, receive, send):
        assert scope["type"] == "http"
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body"):
                break
        self.last_request_body = body
        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": [(b"content-type", b"text/event-stream")],
        })
        for i, chunk in enumerate(self.chunks):
            await send({"type": "http.response.body", "body": chunk, "more_body": i < len(self.chunks) - 1})


@pytest.fixture
def app_with_active_deployment(tmp_path, monkeypatch):
    fake_engine = FakeEngineApp([b"data: hello\n\n", b"data: [DONE]\n\n"])

    # Patch httpx.AsyncClient.stream used by the proxy to talk to the engine
    def fake_async_client_factory(*args, **kwargs):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_engine), base_url="http://engine")
    monkeypatch.setattr("serve_engine.daemon.openai_proxy.make_engine_client", fake_async_client_factory)

    # Skip docker + health check during deployment load
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.wait_healthy", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "serve_engine.lifecycle.manager.download_model_async",
        AsyncMock(return_value=str(tmp_path / "w")),
    )
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="engine", address="engine", port=8000
    )
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    app = build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )

    # Create an active deployment via admin API
    async def setup():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
            r = await c.post("/admin/deployments", json={
                "model_name": "llama-1b",
                "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
                "image_tag": "img:v1",
                "gpu_ids": [0],
                "max_model_len": 8192,
            })
            assert r.status_code == 201
    asyncio.run(setup())
    return app, fake_engine


@pytest.mark.asyncio
async def test_proxy_streams_response(app_with_active_deployment):
    app, fake = app_with_active_deployment
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        async with c.stream(
            "POST", "/v1/chat/completions",
            json={"model": "llama-1b", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as r:
            chunks = [c async for c in r.aiter_bytes()]
    assert r.status_code == 200
    body = b"".join(chunks)
    assert b"hello" in body
    assert b"[DONE]" in body
    forwarded = json.loads(fake.last_request_body)
    assert forwarded["model"] == "llama-1b"


@pytest.mark.asyncio
async def test_proxy_404_when_no_active(tmp_path, monkeypatch):
    docker_client = MagicMock()
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    app = build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/v1/chat/completions", json={"model": "llama-1b", "messages": []})
    assert r.status_code == 503
    assert "no active deployment" in r.json()["detail"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_openai_proxy.py -v`
Expected: FAIL - `openai_proxy` missing.

- [ ] **Step 3: Implement `src/serve_engine/daemon/openai_proxy.py`**

```python
from __future__ import annotations
import sqlite3
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from serve_engine.backends.base import Backend
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store


router = APIRouter()

ENGINE_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)


def make_engine_client(base_url: str) -> httpx.AsyncClient:
    """Factory wrapper so tests can monkeypatch transport."""
    return httpx.AsyncClient(base_url=base_url, timeout=ENGINE_TIMEOUT)


async def _proxy(request: Request, openai_subpath: str) -> StreamingResponse:
    conn: sqlite3.Connection = request.app.state.conn
    backends: dict[str, Backend] = request.app.state.backends

    active = dep_store.find_active(conn)
    if active is None or active.status != "ready":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no active deployment ready to serve",
        )

    backend = backends.get(active.backend)
    if backend is None:
        raise HTTPException(500, detail=f"unknown backend {active.backend!r}")

    base = f"http://{active.container_name}:{active.container_port}{backend.openai_base}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    client = make_engine_client(base)
    upstream = client.stream("POST", openai_subpath, content=body, headers=headers)

    async def streamer():
        try:
            async with upstream as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
        finally:
            await client.aclose()

    # We can't see the upstream status until streaming begins, so we accept whatever it sends.
    return StreamingResponse(streamer(), media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy(request, "/chat/completions")


@router.post("/v1/completions")
async def completions(request: Request):
    return await _proxy(request, "/completions")


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _proxy(request, "/embeddings")


@router.get("/v1/models")
def models(request: Request):
    conn: sqlite3.Connection = request.app.state.conn
    active = dep_store.find_active(conn)
    rows = model_store.list_all(conn)
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "owned_by": "serve-engine",
                "loaded": active is not None and active.model_id == m.id and active.status == "ready",
            }
            for m in rows
        ],
    }
```

- [ ] **Step 4: Mount the proxy router in `src/serve_engine/daemon/app.py`**

Add to the top:
```python
from serve_engine.daemon.openai_proxy import router as openai_router
```
And inside `build_app`, after `app.include_router(admin_router)`:
```python
    app.include_router(openai_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/integration/test_openai_proxy.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/daemon/openai_proxy.py src/serve_engine/daemon/app.py tests/integration/
git commit -m "feat(daemon): OpenAI-compatible streaming proxy to the active engine"
```

---

## Task 12: Daemon entry-point binding to UDS + TCP

**Files:**
- Create: `src/serve_engine/daemon/__main__.py`

The daemon must bind to:
- A TCP socket (`127.0.0.1:11500` by default) for public inference traffic.
- A Unix domain socket (`~/.serve/sock`) for CLI control.

Uvicorn does not accept multiple bindings in one process. We start two uvicorn `Server`s sharing the same FastAPI app, in one asyncio loop.

- [ ] **Step 1: Implement `src/serve_engine/daemon/__main__.py`**

```python
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import structlog
import uvicorn

from serve_engine import config
from serve_engine.backends.vllm import VLLMBackend
from serve_engine.daemon.app import build_app
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.store import db


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


async def serve(public_host: str, public_port: int, sock_path: Path) -> None:
    config.SERVE_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    conn = db.connect(config.DB_PATH)
    db.init_schema(conn)

    docker_client = DockerClient(network_name=config.DOCKER_NETWORK_NAME)
    docker_client.ensure_network()

    app = build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=config.MODELS_DIR,
    )

    if sock_path.exists():
        sock_path.unlink()

    tcp_cfg = uvicorn.Config(app=app, host=public_host, port=public_port, log_level="info")
    uds_cfg = uvicorn.Config(app=app, uds=str(sock_path), log_level="info")
    tcp_server = uvicorn.Server(tcp_cfg)
    uds_server = uvicorn.Server(uds_cfg)
    await asyncio.gather(tcp_server.serve(), uds_server.serve())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="serve-engine-daemon")
    p.add_argument("--host", default=config.DEFAULT_PUBLIC_HOST)
    p.add_argument("--port", default=config.DEFAULT_PUBLIC_PORT, type=int)
    p.add_argument("--sock", default=str(config.SOCK_PATH))
    args = p.parse_args(argv)

    configure_logging()
    asyncio.run(serve(args.host, args.port, Path(args.sock)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Manual smoke test (no automated test in this task)**

Run in one terminal:
```bash
python -m serve_engine.daemon --port 11500
```
Then in another:
```bash
curl http://127.0.0.1:11500/healthz
curl --unix-socket ~/.serve/sock http://localhost/healthz
```
Expected: both return `{"ok": true}`. Stop the daemon with Ctrl-C.

- [ ] **Step 3: Commit**

```bash
git add src/serve_engine/daemon/__main__.py
git commit -m "feat(daemon): entry-point binding to TCP + Unix domain socket"
```

---

## Task 13: CLI IPC client

**Files:**
- Create: `src/serve_engine/cli/__init__.py`
- Create: `src/serve_engine/cli/ipc.py`
- Create: `tests/unit/test_cli_ipc.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli_ipc.py`:
```python
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from serve_engine.cli import ipc


@pytest.mark.asyncio
async def test_ipc_get_uses_uds_transport(tmp_path, monkeypatch):
    sock = tmp_path / "sock"
    captured = {}

    class StubClient:
        def __init__(self, transport, base_url, timeout):
            captured["transport"] = transport
            captured["base_url"] = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, path):
            captured["path"] = path
            return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(ipc.httpx, "AsyncClient", StubClient)
    result = await ipc.get(sock, "/admin/models")
    assert result == {"ok": True}
    assert isinstance(captured["transport"], httpx.AsyncHTTPTransport)
    assert captured["base_url"] == "http://daemon"
    assert captured["path"] == "/admin/models"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli_ipc.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/serve_engine/cli/__init__.py`**

```python
from __future__ import annotations
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="serve - single-node inference orchestrator")
```

- [ ] **Step 4: Implement `src/serve_engine/cli/ipc.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "http://daemon"


def _client(sock: Path) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(uds=str(sock))
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL, timeout=600.0)


async def get(sock: Path, path: str) -> Any:
    async with _client(sock) as c:
        r = await c.get(path)
        r.raise_for_status()
        return r.json()


async def post(sock: Path, path: str, *, json: dict[str, Any] | None = None) -> Any:
    async with _client(sock) as c:
        r = await c.post(path, json=json)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"daemon error {r.status_code}: {detail}")
        if r.status_code == 204:
            return None
        return r.json()


async def delete(sock: Path, path: str) -> None:
    async with _client(sock) as c:
        r = await c.delete(path)
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"daemon error {r.status_code}: {r.text}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_cli_ipc.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/cli/__init__.py src/serve_engine/cli/ipc.py tests/unit/test_cli_ipc.py
git commit -m "feat(cli): Typer app skeleton + UDS-backed IPC client"
```

---

## Task 14: CLI daemon control (start / stop / status)

**Files:**
- Create: `src/serve_engine/cli/daemon_cmd.py`
- Modify: `src/serve_engine/cli/__init__.py` (register)

- [ ] **Step 1: Implement `src/serve_engine/cli/daemon_cmd.py`**

(No unit test in this task - daemon control involves subprocess spawning and process lifecycle which is brittle to test in unit form. Smoke-tested in Task 18.)

```python
from __future__ import annotations
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import typer

from serve_engine import config
from serve_engine.cli import app, ipc

daemon_app = typer.Typer(help="Daemon control")
app.add_typer(daemon_app, name="daemon")

PID_FILE = config.SERVE_DIR / "daemon.pid"


def _is_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@daemon_app.command("start")
def daemon_start(
    host: str = typer.Option(config.DEFAULT_PUBLIC_HOST),
    port: int = typer.Option(config.DEFAULT_PUBLIC_PORT),
):
    """Start the daemon in the background."""
    if _is_running():
        typer.echo("daemon already running")
        raise typer.Exit(0)
    config.SERVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "serve_engine.daemon", "--host", host, "--port", str(port)],
        stdout=open(log_path, "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    # Wait for daemon to be ready
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
            typer.echo(f"daemon started (pid {proc.pid}) on http://{host}:{port}")
            return
        except Exception:
            time.sleep(0.5)
    typer.echo("daemon failed to become ready within 30s", err=True)
    raise typer.Exit(1)


@daemon_app.command("stop")
def daemon_stop():
    """Stop the daemon."""
    if not _is_running():
        typer.echo("daemon not running")
        raise typer.Exit(0)
    pid = int(PID_FILE.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except OSError:
            break
    if PID_FILE.exists():
        PID_FILE.unlink()
    typer.echo("daemon stopped")


@daemon_app.command("status")
def daemon_status():
    """Show daemon status."""
    if not _is_running():
        typer.echo("daemon: not running")
        raise typer.Exit(1)
    pid = int(PID_FILE.read_text().strip())
    try:
        body = asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
        typer.echo(f"daemon: running (pid {pid}), healthz: {body}")
    except Exception as e:
        typer.echo(f"daemon: pid file present (pid {pid}) but unhealthy: {e}", err=True)
        raise typer.Exit(2) from e
```

- [ ] **Step 2: Manual smoke test**

```bash
uv pip install -e .
serve daemon status   # expects: not running
serve daemon start    # expects: started + healthz
serve daemon status   # expects: running
serve daemon stop     # expects: stopped
```

- [ ] **Step 3: Commit**

```bash
git add src/serve_engine/cli/daemon_cmd.py
git commit -m "feat(cli): daemon start/stop/status commands"
```

---

## Task 15: CLI `pull` command

**Files:**
- Create: `src/serve_engine/cli/pull_cmd.py`

The CLI `pull` just registers the model with the daemon; the daemon downloads on first load. (Pre-download as a separate flag is post-MVP.)

- [ ] **Step 1: Implement `src/serve_engine/cli/pull_cmd.py`**

```python
from __future__ import annotations
import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("pull")
def pull(
    hf_repo: str = typer.Argument(..., help="HuggingFace repo id, e.g. meta-llama/Llama-3.2-1B-Instruct"),
    name: str = typer.Option(None, "--name", "-n", help="Local name for the model (default: repo basename)"),
    revision: str = typer.Option("main", "--revision"),
):
    """Register a model with the daemon (download happens at first load)."""
    local_name = name or hf_repo.split("/")[-1].lower()
    body = {"name": local_name, "hf_repo": hf_repo, "revision": revision}
    try:
        result = asyncio.run(ipc.post(config.SOCK_PATH, "/admin/models", json=body))
    except RuntimeError as e:
        typer.echo(f"pull failed: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"registered: {result['name']} ({result['hf_repo']}@{result['revision']})")
```

- [ ] **Step 2: Ensure registration**

Add to `src/serve_engine/cli/__init__.py`:
```python
from serve_engine.cli import pull_cmd  # noqa: F401  registers command
```

- [ ] **Step 3: Manual smoke test**

```bash
serve daemon start
serve pull meta-llama/Llama-3.2-1B-Instruct
# expects: registered: llama-3.2-1b-instruct ...
```

- [ ] **Step 4: Commit**

```bash
git add src/serve_engine/cli/pull_cmd.py src/serve_engine/cli/__init__.py
git commit -m "feat(cli): pull command (model registration)"
```

---

## Task 16: CLI `run` / `stop` / `ps` / `ls` commands

**Files:**
- Create: `src/serve_engine/cli/run_cmd.py`
- Create: `src/serve_engine/cli/stop_cmd.py`
- Create: `src/serve_engine/cli/ps_cmd.py`
- Create: `src/serve_engine/cli/ls_cmd.py`
- Modify: `src/serve_engine/cli/__init__.py`

- [ ] **Step 1: Implement `src/serve_engine/cli/run_cmd.py`**

```python
from __future__ import annotations
import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("run")
def run(
    name_or_repo: str = typer.Argument(..., help="Model name (if registered) or HF repo"),
    gpu: str = typer.Option("0", "--gpu", help="Comma-separated GPU ids, e.g. '0' or '0,1'"),
    max_model_len: int = typer.Option(8192, "--ctx"),
    dtype: str = typer.Option("auto"),
    image_tag: str = typer.Option(None, "--image", help="Override engine image tag"),
):
    """Load a model and make it active. Stops the current model first."""
    gpu_ids = [int(g) for g in gpu.split(",") if g.strip()]
    if "/" in name_or_repo:
        # HF repo path -> register on the fly with derived name
        hf_repo = name_or_repo
        local_name = hf_repo.split("/")[-1].lower()
    else:
        hf_repo = name_or_repo  # The daemon will reject if it's neither registered nor a repo; for v1 require registration first
        local_name = name_or_repo
        # Confirm it's registered
        models = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/models"))
        if not any(m["name"] == local_name for m in models):
            typer.echo(f"model {local_name!r} not registered. Use `serve pull <hf-repo>` first.", err=True)
            raise typer.Exit(1)
        # Look up the HF repo for the registered model
        match = next(m for m in models if m["name"] == local_name)
        hf_repo = match["hf_repo"]

    body = {
        "model_name": local_name,
        "hf_repo": hf_repo,
        "gpu_ids": gpu_ids,
        "max_model_len": max_model_len,
        "dtype": dtype,
    }
    if image_tag is not None:
        body["image_tag"] = image_tag

    typer.echo(f"loading {local_name} on GPU(s) {gpu_ids} ...")
    try:
        result = asyncio.run(ipc.post(config.SOCK_PATH, "/admin/deployments", json=body))
    except RuntimeError as e:
        typer.echo(f"load failed: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"ready: deployment #{result['id']} ({result['container_name']})")
```

- [ ] **Step 2: Implement `src/serve_engine/cli/stop_cmd.py`**

```python
from __future__ import annotations
import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("stop")
def stop():
    """Stop the currently active deployment."""
    asyncio.run(ipc.delete(config.SOCK_PATH, "/admin/deployments/current"))
    typer.echo("stopped")
```

- [ ] **Step 3: Implement `src/serve_engine/cli/ps_cmd.py`**

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
    typer.echo(f"{'ID':<4} {'STATUS':<10} {'BACKEND':<8} {'GPUs':<10} {'CONTAINER':<30}")
    for d in deps:
        typer.echo(
            f"{d['id']:<4} {d['status']:<10} {d['backend']:<8} "
            f"{','.join(str(g) for g in d['gpu_ids']):<10} {d.get('container_name') or '-':<30}"
        )
```

- [ ] **Step 4: Implement `src/serve_engine/cli/ls_cmd.py`**

```python
from __future__ import annotations
import asyncio
import json

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("ls")
def ls(json_out: bool = typer.Option(False, "--json")):
    """List registered models."""
    models = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/models"))
    if json_out:
        typer.echo(json.dumps(models, indent=2))
        return
    if not models:
        typer.echo("no models registered. Use `serve pull <hf-repo>` to add one.")
        return
    typer.echo(f"{'NAME':<30} {'REPO':<50} {'REVISION':<10}")
    for m in models:
        typer.echo(f"{m['name']:<30} {m['hf_repo']:<50} {m['revision']:<10}")
```

- [ ] **Step 5: Register all four in `src/serve_engine/cli/__init__.py`**

```python
from serve_engine.cli import pull_cmd, run_cmd, stop_cmd, ps_cmd, ls_cmd  # noqa: F401
```

- [ ] **Step 6: Commit**

```bash
git add src/serve_engine/cli/run_cmd.py src/serve_engine/cli/stop_cmd.py src/serve_engine/cli/ps_cmd.py src/serve_engine/cli/ls_cmd.py src/serve_engine/cli/__init__.py
git commit -m "feat(cli): run, stop, ps, ls commands"
```

---

## Task 17: CLI `logs` command

**Files:**
- Create: `src/serve_engine/cli/logs_cmd.py`
- Modify: `src/serve_engine/daemon/admin.py` (add `/admin/deployments/current/logs` SSE endpoint)
- Modify: `src/serve_engine/cli/__init__.py`

- [ ] **Step 1: Add a log-streaming admin endpoint to `src/serve_engine/daemon/admin.py`**

Add at the bottom of `admin.py`:
```python
from fastapi.responses import StreamingResponse


@router.get("/deployments/current/logs")
def stream_current_logs(request: Request):
    conn: sqlite3.Connection = request.app.state.conn
    docker_client = request.app.state.manager._docker  # acceptable: same module
    active = dep_store.find_active(conn)
    if active is None or active.container_id is None:
        raise HTTPException(404, "no active deployment with a running container")

    def gen():
        for chunk in docker_client.stream_logs(active.container_id, follow=True):
            if isinstance(chunk, bytes):
                yield chunk
            else:
                yield chunk.encode()

    return StreamingResponse(gen(), media_type="text/plain")
```

(Reaching into `manager._docker` is acceptable here because Plan 02 promotes a public accessor as part of its refactor; we don't pre-build that.)

- [ ] **Step 2: Implement `src/serve_engine/cli/logs_cmd.py`**

```python
from __future__ import annotations
import asyncio

import httpx
import typer

from serve_engine import config
from serve_engine.cli import app


@app.command("logs")
def logs(follow: bool = typer.Option(True, "--follow/--no-follow", "-f")):
    """Stream logs from the currently active deployment's engine container."""
    async def run():
        transport = httpx.AsyncHTTPTransport(uds=str(config.SOCK_PATH))
        async with httpx.AsyncClient(transport=transport, base_url="http://daemon", timeout=None) as c:
            try:
                async with c.stream("GET", "/admin/deployments/current/logs") as r:
                    if r.status_code != 200:
                        typer.echo(await r.aread(), err=True)
                        raise typer.Exit(1)
                    async for chunk in r.aiter_raw():
                        typer.echo(chunk.decode(errors="replace"), nl=False)
            except KeyboardInterrupt:
                pass
    asyncio.run(run())
```

- [ ] **Step 3: Register in `src/serve_engine/cli/__init__.py`**

```python
from serve_engine.cli import pull_cmd, run_cmd, stop_cmd, ps_cmd, ls_cmd, logs_cmd  # noqa: F401
```

- [ ] **Step 4: Commit**

```bash
git add src/serve_engine/cli/logs_cmd.py src/serve_engine/cli/__init__.py src/serve_engine/daemon/admin.py
git commit -m "feat(cli): logs command streaming from active container"
```

---

## Task 18: End-to-end smoke-test script (manual, no CI GPU)

**Files:**
- Create: `scripts/smoke_e2e.sh`

This is a documented manual test for a developer with Docker + nvidia-container-toolkit + a small model accessible. It is not run in CI.

- [ ] **Step 1: Write `scripts/smoke_e2e.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Prereqs:
#   - Docker daemon running
#   - nvidia-container-toolkit configured
#   - At least one CUDA-capable GPU (id 0)
#   - HuggingFace cache primed or HF_TOKEN set if model is gated
#   - `uv pip install -e ".[dev]"` already done
#
# What this verifies end-to-end:
#   - daemon start/stop
#   - pull (register)
#   - run (download + container spawn + health-wait)
#   - /v1/chat/completions returns a streamed body
#   - stop tears down the container

cleanup() {
    serve stop 2>/dev/null || true
    serve daemon stop 2>/dev/null || true
}
trap cleanup EXIT

serve daemon start

serve pull meta-llama/Llama-3.2-1B-Instruct --name llama-1b

serve run llama-1b --gpu 0 --ctx 8192 --dtype auto

echo "Hitting /v1/chat/completions ..."
curl -sS -N http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-1b",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "stream": true,
    "max_tokens": 4
  }' | tee /tmp/serve_smoke.out

if ! grep -q "OK" /tmp/serve_smoke.out; then
    echo "FAIL: expected 'OK' in streamed response"
    exit 1
fi

echo "PASS"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/smoke_e2e.sh
git add scripts/smoke_e2e.sh
git commit -m "test: end-to-end manual smoke script"
```

---

## Task 19: README quickstart

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# serve-engine

A single-node, multi-user inference orchestrator over vLLM (and soon SGLang).

Goal: solve the operator UX gap left by `vllm serve` / `python -m sglang.launch_server` - one daemon, multi-model, OpenAI-compatible, container-isolated engines, no YAML.

This repository is in active early development. Plan 01 (the walking skeleton) ships single-model serving over vLLM.

## Requirements

- Linux with NVIDIA GPU(s)
- Docker 24+ with `nvidia-container-toolkit`
- Python 3.11+ and `uv`

## Quickstart

```bash
# Install
uv tool install -e .

# Start the daemon
serve daemon start

# Register a model
serve pull meta-llama/Llama-3.2-1B-Instruct --name llama-1b

# Load it on GPU 0
serve run llama-1b --gpu 0

# Talk to it (any OpenAI client works)
curl http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-1b","messages":[{"role":"user","content":"Hello"}]}'

# Stop it
serve stop
serve daemon stop
```

## Plan 01 limitations (intentional)

- One model at a time. Loading a new model stops the current one.
- vLLM only. SGLang comes in Plan 04.
- No authentication. The TCP port is open to anyone on the host; bind to `127.0.0.1` (default) and use a reverse proxy if exposing externally.
- No autotune. You pass `--gpu`, `--ctx`, `--dtype` yourself.
- No web UI.

See `docs/design/specs/` for the full design and `docs/design/plans/` for the planned slices.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README quickstart for Plan 01"
```

---

## Verification (end of Plan 01)

After all tasks are complete, the following must hold:

1. `pytest tests/ -v` - all unit + integration tests pass with no GPU.
2. `ruff check src/ tests/` - clean.
3. `mypy src/serve_engine` - clean (or documented exceptions).
4. On a real machine with Docker + nvidia-container-toolkit + a GPU:
   - `bash scripts/smoke_e2e.sh` exits 0 and prints `PASS`.

If any of these fails, the plan is not complete.

---

## Self-review

**Spec coverage check (Plan 01 scope only):**

- Daemon process bound to UDS + TCP - Task 12 OK
- CLI as thin client over UDS - Tasks 13-17 OK
- Per-model container via Docker API (upstream `vllm/vllm-openai`) - Tasks 6, 7, 9 OK
- Container on `serve-engines` bridge network, addressed by name - Task 7 OK
- Health-check before marking ready - Task 9 OK
- OpenAI-compatible proxy with streaming - Task 11 OK
- Model registry + on-first-load HF download - Tasks 3, 8, 9 OK
- SQLite state (models + deployments) - Tasks 2, 3, 4 OK
- Failure handling (engine unhealthy -> cleanup + failed status) - Task 9 OK

**Explicitly deferred (correct per Plan 01 scope):**

- Multi-deployment lifecycle, pin/auto-swap - Plan 02
- GPU topology / placement - Plan 02
- Autotune - Plan 03
- SGLang backend, engine selection, `backends.yaml` - Plan 04
- API keys, fair queueing, multi-tenancy - Plan 05
- `/metrics` aggregation, `/admin/events`, `serve top` - Plan 06
- Web UI - Plan 07
- `serve doctor`, install script, daemon-as-container - Plan 08

**Placeholder scan:** none - every code step contains the actual code, every command step the actual command.

**Type consistency check:** `DeploymentPlan` fields used identically across `vllm.py`, `manager.py`, `admin.py`. `Deployment` dataclass fields used identically across `deployments.py`, `admin.py`. `ContainerHandle` used identically across `docker_client.py` and `manager.py`. No mismatches found.
