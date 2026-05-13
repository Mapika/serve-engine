# Serving Engine — Plan 04: API Keys + Multi-Window Rate Limits

**Goal:** Bearer-token auth with proper rate limits at minute / hour / day / week scales (RPM, RPH, RPD, RPW for requests and TPM, TPH, TPD, TPW for tokens), matching what OpenAI / Anthropic / other inference providers expose. Returns `429 Too Many Requests` with `Retry-After` when a limit is hit.

**Architecture:** A `key_store` table for hashed keys, a `key_usage_events` table that records `(key_id, ts, requests, tokens_in, tokens_out)` per request. The rate limiter is a sliding-window log over `key_usage_events` — straightforward to implement, simple to reason about, and fast enough for tens of QPS on SQLite. A FastAPI dependency wraps every `/v1/*` route. Admin endpoints + CLI to manage keys and tier definitions. Tiers live in YAML so a homelab can run with one tier ("admin") while a company defines many.

**Tech Stack:** Same as Plans 01–03. New: nothing — sha256 hashing from stdlib, sqlite for usage log, fastapi dependency.

**Backward compat:** If no keys are registered (`api_keys` table empty), auth is bypassed entirely. This preserves Plan 01–03 "trust your network" UX for homelab users until they explicitly opt in by creating their first key.

---

## File structure

```
serving-engine/
├── src/serve_engine/
│   ├── store/
│   │   ├── migrations/003_api_keys.sql      # NEW
│   │   ├── api_keys.py                      # NEW
│   │   └── key_usage.py                     # NEW
│   ├── auth/
│   │   ├── __init__.py                      # NEW (empty)
│   │   ├── tiers.yaml                       # NEW — packaged tier presets
│   │   ├── tiers.py                         # NEW — load tier limits
│   │   ├── limiter.py                       # NEW — sliding-window logic
│   │   └── middleware.py                    # NEW — FastAPI dependency
│   ├── daemon/
│   │   ├── admin.py                         # MODIFIED — /admin/keys, /admin/usage
│   │   ├── openai_proxy.py                  # MODIFIED — auth dep + token tracking
│   │   └── app.py                           # MODIFIED — mount auth dep
│   └── cli/
│       ├── key_cmd.py                       # NEW — serve key {create|list|revoke|show}
│       └── __init__.py                      # MODIFIED — register key_cmd
└── tests/
    └── unit/
        ├── test_api_keys_store.py           # NEW
        ├── test_key_usage_store.py          # NEW
        ├── test_tiers.py                    # NEW
        ├── test_limiter.py                  # NEW
        └── test_auth_middleware.py          # NEW
```

---

## Task 1: Schema migration 003

**Files:**
- Create: `src/serve_engine/store/migrations/003_api_keys.sql`

- [ ] **Step 1: Write the migration**

```sql
-- Plan 04: API keys + per-key usage events.

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                           -- human-readable label
    prefix TEXT NOT NULL,                         -- "sk-aBc..." first 8 chars, for listing
    key_hash TEXT NOT NULL UNIQUE,                -- sha256 of full secret
    tier TEXT NOT NULL DEFAULT 'standard',
    -- Optional per-key overrides (NULL → use tier defaults)
    rpm_override INTEGER,
    tpm_override INTEGER,
    rpd_override INTEGER,
    tpd_override INTEGER,
    rph_override INTEGER,
    tph_override INTEGER,
    rpw_override INTEGER,
    tpw_override INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP,
    last_used_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked_at);

CREATE TABLE IF NOT EXISTS key_usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    model_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_key_usage_events_key_ts ON key_usage_events(key_id, ts);
CREATE INDEX IF NOT EXISTS idx_key_usage_events_ts ON key_usage_events(ts);
```

- [ ] **Step 2: Commit**

```bash
git add src/serve_engine/store/migrations/003_api_keys.sql
git commit -m "feat(store): schema 003 — api_keys + key_usage_events"
```

(No tests yet — Task 2 introduces the consumer.)

---

## Task 2: api_keys store

**Files:**
- Create: `src/serve_engine/store/api_keys.py`
- Create: `tests/unit/test_api_keys_store.py`

Keys are SHA-256 hashed (with the OpenAI-style `sk-` prefix preserved for compatibility). Verification is constant-time. Listing returns the prefix-only form so the UI can show `sk-abc1...` without reconstructing the secret.

- [ ] **Step 1: Write tests**

`tests/unit/test_api_keys_store.py`:
```python
import pytest

from serve_engine.store import api_keys, db


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def test_create_returns_full_key_once(tmp_path):
    conn = _fresh(tmp_path)
    secret, key = api_keys.create(conn, name="alice", tier="standard")
    assert secret.startswith("sk-")
    assert len(secret) >= 32  # plausible entropy
    assert key.prefix == secret[:12]
    assert key.name == "alice"
    assert key.tier == "standard"


def test_verify_matches_hashed(tmp_path):
    conn = _fresh(tmp_path)
    secret, _ = api_keys.create(conn, name="alice", tier="standard")
    found = api_keys.verify(conn, secret)
    assert found is not None
    assert found.name == "alice"


def test_verify_rejects_wrong_secret(tmp_path):
    conn = _fresh(tmp_path)
    api_keys.create(conn, name="alice", tier="standard")
    assert api_keys.verify(conn, "sk-wrong") is None


def test_verify_rejects_revoked(tmp_path):
    conn = _fresh(tmp_path)
    secret, key = api_keys.create(conn, name="alice", tier="standard")
    api_keys.revoke(conn, key.id)
    assert api_keys.verify(conn, secret) is None


def test_list_all_excludes_secret(tmp_path):
    conn = _fresh(tmp_path)
    api_keys.create(conn, name="a", tier="standard")
    api_keys.create(conn, name="b", tier="standard")
    rows = api_keys.list_all(conn)
    assert len(rows) == 2
    assert all(not hasattr(r, "secret") for r in rows)


def test_per_key_overrides(tmp_path):
    conn = _fresh(tmp_path)
    _, key = api_keys.create(
        conn, name="alice", tier="standard", rpm_override=120, tpm_override=50_000,
    )
    fetched = api_keys.get_by_id(conn, key.id)
    assert fetched.rpm_override == 120
    assert fetched.tpm_override == 50_000
    assert fetched.rpd_override is None
```

Run `pytest tests/unit/test_api_keys_store.py -v` → FAIL (module missing).

- [ ] **Step 2: Implement `src/serve_engine/store/api_keys.py`**

```python
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiKey:
    id: int
    name: str
    prefix: str
    tier: str
    rpm_override: int | None
    tpm_override: int | None
    rpd_override: int | None
    tpd_override: int | None
    rph_override: int | None
    tph_override: int | None
    rpw_override: int | None
    tpw_override: int | None
    revoked_at: str | None


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _row_to_key(row: sqlite3.Row) -> ApiKey:
    return ApiKey(
        id=row["id"],
        name=row["name"],
        prefix=row["prefix"],
        tier=row["tier"],
        rpm_override=row["rpm_override"],
        tpm_override=row["tpm_override"],
        rpd_override=row["rpd_override"],
        tpd_override=row["tpd_override"],
        rph_override=row["rph_override"],
        tph_override=row["tph_override"],
        rpw_override=row["rpw_override"],
        tpw_override=row["tpw_override"],
        revoked_at=row["revoked_at"],
    )


def create(
    conn: sqlite3.Connection,
    *,
    name: str,
    tier: str = "standard",
    rpm_override: int | None = None,
    tpm_override: int | None = None,
    rpd_override: int | None = None,
    tpd_override: int | None = None,
    rph_override: int | None = None,
    tph_override: int | None = None,
    rpw_override: int | None = None,
    tpw_override: int | None = None,
) -> tuple[str, ApiKey]:
    """Generate a new key. Returns (secret, ApiKey). The secret is only available here."""
    body = secrets.token_urlsafe(32)
    secret = f"sk-{body}"
    prefix = secret[:12]
    key_hash = _hash(secret)
    cur = conn.execute(
        """
        INSERT INTO api_keys
            (name, prefix, key_hash, tier,
             rpm_override, tpm_override, rpd_override, tpd_override,
             rph_override, tph_override, rpw_override, tpw_override)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name, prefix, key_hash, tier,
            rpm_override, tpm_override, rpd_override, tpd_override,
            rph_override, tph_override, rpw_override, tpw_override,
        ),
    )
    fetched = get_by_id(conn, cur.lastrowid)
    assert fetched is not None
    return secret, fetched


def get_by_id(conn: sqlite3.Connection, key_id: int) -> ApiKey | None:
    row = conn.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
    return _row_to_key(row) if row else None


def verify(conn: sqlite3.Connection, secret: str) -> ApiKey | None:
    """Look up a key by secret; returns None if missing or revoked."""
    candidate_hash = _hash(secret)
    row = conn.execute(
        "SELECT * FROM api_keys WHERE key_hash=? AND revoked_at IS NULL",
        (candidate_hash,),
    ).fetchone()
    if row is None:
        return None
    # Constant-time check to avoid timing oracles (sqlite already matched, but be explicit)
    if not hmac.compare_digest(row["key_hash"], candidate_hash):
        return None
    conn.execute(
        "UPDATE api_keys SET last_used_at=CURRENT_TIMESTAMP WHERE id=?",
        (row["id"],),
    )
    return _row_to_key(row)


def list_all(conn: sqlite3.Connection) -> list[ApiKey]:
    rows = conn.execute(
        "SELECT * FROM api_keys ORDER BY id"
    ).fetchall()
    return [_row_to_key(r) for r in rows]


def revoke(conn: sqlite3.Connection, key_id: int) -> None:
    conn.execute(
        "UPDATE api_keys SET revoked_at=CURRENT_TIMESTAMP WHERE id=?",
        (key_id,),
    )


def count_active(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM api_keys WHERE revoked_at IS NULL"
    ).fetchone()
    return int(row["n"])
```

- [ ] **Step 3: Run tests + commit**

```bash
pytest tests/unit/test_api_keys_store.py -v
ruff check src/ tests/
git add src/serve_engine/store/api_keys.py tests/unit/test_api_keys_store.py
git commit -m "feat(store): api_keys CRUD with sha256 hashing"
```

---

## Task 3: key_usage_events store

**Files:**
- Create: `src/serve_engine/store/key_usage.py`
- Create: `tests/unit/test_key_usage_store.py`

- [ ] **Step 1: Tests**

`tests/unit/test_key_usage_store.py`:
```python
import time

from serve_engine.store import api_keys, db, key_usage


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def test_record_and_count_in_window(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="standard")
    key_usage.record(conn, key_id=k.id, tokens_in=100, tokens_out=50, model_name="qwen-0_5b")
    key_usage.record(conn, key_id=k.id, tokens_in=10, tokens_out=20)
    requests, tokens = key_usage.totals_in_window(conn, key_id=k.id, window_s=60)
    assert requests == 2
    assert tokens == 100 + 50 + 10 + 20


def test_purge_older_than(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="standard")
    key_usage.record(conn, key_id=k.id, tokens_in=1, tokens_out=1)
    time.sleep(1.1)
    key_usage.record(conn, key_id=k.id, tokens_in=2, tokens_out=2)
    # Purge anything older than 0.5 s
    purged = key_usage.purge_older_than_s(conn, max_age_s=0.5)
    assert purged == 1
    requests, _ = key_usage.totals_in_window(conn, key_id=k.id, window_s=60)
    assert requests == 1
```

- [ ] **Step 2: Implement `src/serve_engine/store/key_usage.py`**

```python
from __future__ import annotations

import sqlite3


def record(
    conn: sqlite3.Connection,
    *,
    key_id: int,
    tokens_in: int,
    tokens_out: int,
    model_name: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO key_usage_events (key_id, tokens_in, tokens_out, model_name)
        VALUES (?, ?, ?, ?)
        """,
        (key_id, tokens_in, tokens_out, model_name),
    )


def totals_in_window(
    conn: sqlite3.Connection,
    *,
    key_id: int,
    window_s: int,
) -> tuple[int, int]:
    """Returns (request_count, total_tokens) for events with ts > now - window_s."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            COALESCE(SUM(tokens_in + tokens_out), 0) AS tok
        FROM key_usage_events
        WHERE key_id = ?
          AND ts > datetime('now', ?)
        """,
        (key_id, f"-{window_s} seconds"),
    ).fetchone()
    return int(row["n"]), int(row["tok"])


def purge_older_than_s(conn: sqlite3.Connection, *, max_age_s: float) -> int:
    """Delete usage events older than `max_age_s` seconds. Returns rows deleted."""
    cur = conn.execute(
        "DELETE FROM key_usage_events WHERE ts < datetime('now', ?)",
        (f"-{max_age_s} seconds",),
    )
    return cur.rowcount
```

- [ ] **Step 3: Run tests + commit**

```bash
pytest tests/unit/test_key_usage_store.py -v
ruff check src/ tests/
git add src/serve_engine/store/key_usage.py tests/unit/test_key_usage_store.py
git commit -m "feat(store): key_usage_events with sliding-window queries"
```

---

## Task 4: Tier definitions (YAML)

**Files:**
- Create: `src/serve_engine/auth/__init__.py` (empty)
- Create: `src/serve_engine/auth/tiers.yaml`
- Create: `src/serve_engine/auth/tiers.py`
- Create: `tests/unit/test_tiers.py`

Tiers express the *defaults* for a class of keys. Per-key overrides (from Task 2) take precedence.

- [ ] **Step 1: `src/serve_engine/auth/tiers.yaml`**

```yaml
# Default tier definitions. Overridable per-tier via /admin/tiers (not exposed
# in Plan 04; future plans). All values are integer caps per window. null = no cap.

tiers:
  admin:
    rpm: null
    tpm: null
    rpd: null
    tpd: null
    rph: null
    tph: null
    rpw: null
    tpw: null

  standard:
    rpm: 60
    tpm: 100000
    rph: 2000
    tph: 3000000
    rpd: 20000
    tpd: 30000000
    rpw: 100000
    tpw: 200000000

  trial:
    rpm: 10
    tpm: 10000
    rph: 200
    tph: 200000
    rpd: 1000
    tpd: 1000000
    rpw: 5000
    tpw: 5000000
```

- [ ] **Step 2: Tests**

`tests/unit/test_tiers.py`:
```python
import pytest

from serve_engine.auth import tiers


def test_load_default_tiers_contains_standard():
    cfg = tiers.load_tiers()
    assert "standard" in cfg
    assert cfg["standard"].rpm == 60
    assert cfg["admin"].rpm is None  # unlimited


def test_resolve_limits_no_override():
    cfg = tiers.load_tiers()
    limits = tiers.resolve_limits(cfg, tier="standard", overrides=tiers.Overrides())
    assert limits.rpm == 60
    assert limits.tpm == 100_000


def test_resolve_limits_with_override():
    cfg = tiers.load_tiers()
    limits = tiers.resolve_limits(
        cfg, tier="standard",
        overrides=tiers.Overrides(rpm=200, tpm=None),
    )
    assert limits.rpm == 200       # overridden
    assert limits.tpm == 100_000   # falls through


def test_unknown_tier_raises():
    cfg = tiers.load_tiers()
    with pytest.raises(KeyError):
        tiers.resolve_limits(cfg, tier="legendary", overrides=tiers.Overrides())
```

- [ ] **Step 3: `src/serve_engine/auth/__init__.py`** (empty file)

- [ ] **Step 4: `src/serve_engine/auth/tiers.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Limits:
    rpm: int | None
    tpm: int | None
    rph: int | None
    tph: int | None
    rpd: int | None
    tpd: int | None
    rpw: int | None
    tpw: int | None


@dataclass(frozen=True)
class Overrides:
    rpm: int | None = None
    tpm: int | None = None
    rph: int | None = None
    tph: int | None = None
    rpd: int | None = None
    tpd: int | None = None
    rpw: int | None = None
    tpw: int | None = None


def load_tiers(path: Path | None = None) -> dict[str, Limits]:
    if path is None:
        text = files("serve_engine.auth").joinpath("tiers.yaml").read_text()
    else:
        text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    out: dict[str, Limits] = {}
    for name, lim in (data.get("tiers") or {}).items():
        out[name] = Limits(
            rpm=lim.get("rpm"), tpm=lim.get("tpm"),
            rph=lim.get("rph"), tph=lim.get("tph"),
            rpd=lim.get("rpd"), tpd=lim.get("tpd"),
            rpw=lim.get("rpw"), tpw=lim.get("tpw"),
        )
    return out


def resolve_limits(
    cfg: dict[str, Limits], *, tier: str, overrides: Overrides
) -> Limits:
    if tier not in cfg:
        raise KeyError(f"unknown tier {tier!r}")
    base = cfg[tier]
    def pick(o: int | None, b: int | None) -> int | None:
        # An override of `None` does NOT mean unlimited; it means "fall through to tier".
        # A tier `null` is unlimited.
        return o if o is not None else b
    return Limits(
        rpm=pick(overrides.rpm, base.rpm),
        tpm=pick(overrides.tpm, base.tpm),
        rph=pick(overrides.rph, base.rph),
        tph=pick(overrides.tph, base.tph),
        rpd=pick(overrides.rpd, base.rpd),
        tpd=pick(overrides.tpd, base.tpd),
        rpw=pick(overrides.rpw, base.rpw),
        tpw=pick(overrides.tpw, base.tpw),
    )
```

- [ ] **Step 5: Run tests + commit**

```bash
pytest tests/unit/test_tiers.py -v
ruff check src/ tests/
git add src/serve_engine/auth/ tests/unit/test_tiers.py
git commit -m "feat(auth): tier YAML with admin/standard/trial presets + Overrides"
```

---

## Task 5: Rate limiter

**Files:**
- Create: `src/serve_engine/auth/limiter.py`
- Create: `tests/unit/test_limiter.py`

For each request, check all 8 windows (RPM/TPM/RPH/TPH/RPD/TPD/RPW/TPW). Returns either `Allowed()` or `Denied(window_name, retry_after_s)`. The hottest windows (RPM/TPM) are checked first.

- [ ] **Step 1: Tests**

`tests/unit/test_limiter.py`:
```python
import pytest

from serve_engine.auth import limiter, tiers
from serve_engine.store import api_keys, db, key_usage


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def test_allow_when_no_history(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="trial")
    cfg = tiers.load_tiers()
    decision = limiter.check(conn, key=k, tier_cfg=cfg)
    assert isinstance(decision, limiter.Allowed)


def test_deny_when_rpm_exceeded(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="trial")  # rpm=10
    cfg = tiers.load_tiers()
    for _ in range(10):
        key_usage.record(conn, key_id=k.id, tokens_in=1, tokens_out=0)
    decision = limiter.check(conn, key=k, tier_cfg=cfg)
    assert isinstance(decision, limiter.Denied)
    assert decision.limit_name == "rpm"


def test_deny_when_tpm_exceeded(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="trial")  # tpm=10000
    cfg = tiers.load_tiers()
    key_usage.record(conn, key_id=k.id, tokens_in=10000, tokens_out=1)
    decision = limiter.check(conn, key=k, tier_cfg=cfg)
    assert isinstance(decision, limiter.Denied)
    assert decision.limit_name == "tpm"


def test_admin_tier_unlimited(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="admin")
    cfg = tiers.load_tiers()
    for _ in range(10_000):
        key_usage.record(conn, key_id=k.id, tokens_in=1_000_000, tokens_out=0)
    decision = limiter.check(conn, key=k, tier_cfg=cfg)
    assert isinstance(decision, limiter.Allowed)


def test_per_key_override_loosens(tmp_path):
    conn = _fresh(tmp_path)
    _, k = api_keys.create(conn, name="a", tier="trial", rpm_override=100)
    cfg = tiers.load_tiers()
    for _ in range(50):
        key_usage.record(conn, key_id=k.id, tokens_in=1, tokens_out=0)
    decision = limiter.check(conn, key=k, tier_cfg=cfg)
    assert isinstance(decision, limiter.Allowed)
```

- [ ] **Step 2: Implement `src/serve_engine/auth/limiter.py`**

```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from serve_engine.auth.tiers import Limits, Overrides, resolve_limits
from serve_engine.store import api_keys, key_usage


@dataclass(frozen=True)
class Allowed:
    pass


@dataclass(frozen=True)
class Denied:
    limit_name: str
    limit_value: int
    current: int
    window_s: int
    retry_after_s: int


Decision = Allowed | Denied


# (limit_attr, kind, window_s, label)  kind ∈ {"req","tok"}
_WINDOWS = [
    ("rpm", "req", 60, "rpm"),
    ("tpm", "tok", 60, "tpm"),
    ("rph", "req", 3600, "rph"),
    ("tph", "tok", 3600, "tph"),
    ("rpd", "req", 86_400, "rpd"),
    ("tpd", "tok", 86_400, "tpd"),
    ("rpw", "req", 604_800, "rpw"),
    ("tpw", "tok", 604_800, "tpw"),
]


def check(
    conn: sqlite3.Connection,
    *,
    key: api_keys.ApiKey,
    tier_cfg: dict[str, Limits],
) -> Decision:
    overrides = Overrides(
        rpm=key.rpm_override, tpm=key.tpm_override,
        rph=key.rph_override, tph=key.tph_override,
        rpd=key.rpd_override, tpd=key.tpd_override,
        rpw=key.rpw_override, tpw=key.tpw_override,
    )
    limits = resolve_limits(tier_cfg, tier=key.tier, overrides=overrides)

    for attr, kind, window_s, label in _WINDOWS:
        limit = getattr(limits, attr)
        if limit is None:
            continue
        reqs, toks = key_usage.totals_in_window(conn, key_id=key.id, window_s=window_s)
        current = reqs if kind == "req" else toks
        if current >= limit:
            return Denied(
                limit_name=label,
                limit_value=limit,
                current=current,
                window_s=window_s,
                retry_after_s=_retry_after(window_s),
            )
    return Allowed()


def _retry_after(window_s: int) -> int:
    # Coarse estimate: a sliding-window log returns to under-limit when the oldest
    # event in the window expires. Without that timestamp we expose `window_s / 2`
    # as a guess. Plan 05 can refine with the actual oldest-event time.
    return max(1, window_s // 2)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/test_limiter.py -v
ruff check src/ tests/
git add src/serve_engine/auth/limiter.py tests/unit/test_limiter.py
git commit -m "feat(auth): sliding-window rate limiter (8 windows: RPM/TPM × M/H/D/W)"
```

---

## Task 6: FastAPI auth middleware (dependency)

**Files:**
- Create: `src/serve_engine/auth/middleware.py`
- Create: `tests/unit/test_auth_middleware.py`

Implemented as a FastAPI dependency rather than ASGI middleware: this gives us route-level granularity and request.app.state access.

- [ ] **Step 1: Tests**

`tests/unit/test_auth_middleware.py`:
```python
import httpx
import pytest
from fastapi import FastAPI, Depends

from serve_engine.auth.middleware import require_auth_dep
from serve_engine.auth import tiers
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
        async def _test(_ = Depends(require_auth_dep)):
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
```

- [ ] **Step 2: Implement `src/serve_engine/auth/middleware.py`**

```python
from __future__ import annotations

import sqlite3

from fastapi import HTTPException, Request, status

from serve_engine.auth import limiter
from serve_engine.auth.tiers import Limits
from serve_engine.store import api_keys


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def require_auth_dep(request: Request) -> api_keys.ApiKey | None:
    """FastAPI dependency. Returns the ApiKey on success, raises 401/429 on failure.

    If no keys exist in the table, auth is bypassed (returns None).
    """
    conn: sqlite3.Connection = request.app.state.conn
    if api_keys.count_active(conn) == 0:
        return None

    auth_header = request.headers.get("authorization")
    secret = _extract_bearer(auth_header)
    if secret is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header (expected: Bearer sk-...)",
            headers={"WWW-Authenticate": 'Bearer realm="serve-engine"'},
        )

    key = api_keys.verify(conn, secret)
    if key is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="invalid or revoked API key",
        )

    tier_cfg: dict[str, Limits] = request.app.state.tier_cfg
    decision = limiter.check(conn, key=key, tier_cfg=tier_cfg)
    if isinstance(decision, limiter.Denied):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"{decision.limit_name} limit reached "
                f"({decision.current}/{decision.limit_value} in {decision.window_s}s)"
            ),
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    return key
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/unit/test_auth_middleware.py -v
ruff check src/ tests/
git add src/serve_engine/auth/middleware.py tests/unit/test_auth_middleware.py
git commit -m "feat(auth): FastAPI auth dependency — Bearer + tier rate limits + 429 Retry-After"
```

---

## Task 7: Wire auth dep into OpenAI proxy + record usage

**Files:**
- Modify: `src/serve_engine/daemon/openai_proxy.py`
- Modify: `src/serve_engine/daemon/app.py`
- Modify: `tests/integration/test_openai_proxy.py`

Three things happen here:

1. Every `/v1/*` route depends on `require_auth_dep` so unauthenticated requests get 401 (when keys exist).
2. After the engine response streams, we record usage in `key_usage_events`. We extract `tokens_in` and `tokens_out` either from the engine's response JSON (`usage` field for non-streaming) or by lightly parsing the SSE stream.
3. `app.state.tier_cfg` must be populated at startup.

- [ ] **Step 1: In `daemon/app.py` `_attach_state`, populate `tier_cfg`**

Add at the top:
```python
from serve_engine.auth.tiers import load_tiers
```

In `_attach_state`, after setting `app.state.manager`, add:
```python
    app.state.tier_cfg = load_tiers()
```

- [ ] **Step 2: Modify `_proxy` in `daemon/openai_proxy.py`**

Add imports:
```python
from fastapi import Depends
from serve_engine.auth.middleware import require_auth_dep
from serve_engine.store import api_keys as _api_keys_store
from serve_engine.store import key_usage as _key_usage_store
```

Change the route handlers to inject the dependency:

```python
@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    key: _api_keys_store.ApiKey | None = Depends(require_auth_dep),
):
    return await _proxy(request, "/chat/completions", key=key)
```

Apply the same shape to `/v1/completions`, `/v1/embeddings`.

(For `GET /v1/models`, also wrap with the dep but it's a read-only listing — fine to require auth too.)

In `_proxy`, accept the key and record usage after the upstream stream completes. Update the body to:

```python
async def _proxy(
    request: Request,
    openai_subpath: str,
    *,
    key: _api_keys_store.ApiKey | None,
) -> StreamingResponse:
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
    # Strip the user's Authorization (don't leak our API key to the engine).
    headers.pop("authorization", None)
    headers.pop("Authorization", None)

    client = make_engine_client(base)
    upstream = client.stream("POST", openai_subpath, content=body, headers=headers)

    captured = bytearray()

    async def streamer():
        try:
            async with upstream as resp:
                async for chunk in resp.aiter_raw():
                    captured.extend(chunk)
                    yield chunk
        finally:
            await client.aclose()
            if key is not None:
                tin, tout = _extract_usage(bytes(captured))
                _key_usage_store.record(
                    conn, key_id=key.id, tokens_in=tin, tokens_out=tout,
                    model_name=model_name,
                )

    return StreamingResponse(streamer(), media_type="text/event-stream")


def _extract_usage(body: bytes) -> tuple[int, int]:
    """Best-effort token-count extraction from OpenAI-format response or SSE."""
    if not body:
        return 0, 0
    text = body.decode(errors="replace")
    # Try non-streaming JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            usage = obj.get("usage") or {}
            return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    except json.JSONDecodeError:
        pass
    # SSE: vLLM emits a final `data: { ... "usage": {...} ...}` chunk for streams
    # that opt in via stream_options.include_usage=true. If absent, return (0, 0)
    # — we don't introspect every chunk for tokens to avoid hot-path cost.
    for line in reversed(text.splitlines()):
        if line.startswith("data:") and "usage" in line:
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
                usage = obj.get("usage") or {}
                return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
            except json.JSONDecodeError:
                continue
    return 0, 0
```

- [ ] **Step 3: Update existing integration tests for tier_cfg**

In `tests/integration/test_openai_proxy.py`, the existing `app_with_active_deployment` fixture must set `tier_cfg`. The `build_app` call already sets `app.state.tier_cfg` via Task 7 Step 1, so existing tests pass as-is (no keys → bypass).

- [ ] **Step 4: Run + commit**

```bash
pytest -v
ruff check src/ tests/
git add src/serve_engine/daemon/openai_proxy.py src/serve_engine/daemon/app.py tests/integration/test_openai_proxy.py
git commit -m "feat(daemon): auth dep on /v1/* + record key usage after stream"
```

---

## Task 8: Admin endpoints + CLI for keys

**Files:**
- Modify: `src/serve_engine/daemon/admin.py`
- Create: `src/serve_engine/cli/key_cmd.py`
- Modify: `src/serve_engine/cli/__init__.py`
- Modify: `tests/unit/test_admin_endpoints.py`

- [ ] **Step 1: Admin endpoints in `admin.py`**

Append to `admin.py`:

```python
from serve_engine.store import api_keys as _ak_store


class CreateKeyRequest(BaseModel):
    name: str
    tier: str = "standard"
    rpm_override: int | None = None
    tpm_override: int | None = None
    rph_override: int | None = None
    tph_override: int | None = None
    rpd_override: int | None = None
    tpd_override: int | None = None
    rpw_override: int | None = None
    tpw_override: int | None = None


@router.get("/keys")
def list_keys(conn: sqlite3.Connection = Depends(get_conn)):
    return [
        {
            "id": k.id,
            "name": k.name,
            "prefix": k.prefix,
            "tier": k.tier,
            "revoked": k.revoked_at is not None,
        }
        for k in _ak_store.list_all(conn)
    ]


@router.post("/keys", status_code=status.HTTP_201_CREATED)
def create_key(
    body: CreateKeyRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    secret, k = _ak_store.create(
        conn, name=body.name, tier=body.tier,
        rpm_override=body.rpm_override, tpm_override=body.tpm_override,
        rph_override=body.rph_override, tph_override=body.tph_override,
        rpd_override=body.rpd_override, tpd_override=body.tpd_override,
        rpw_override=body.rpw_override, tpw_override=body.tpw_override,
    )
    return {
        "id": k.id,
        "name": k.name,
        "prefix": k.prefix,
        "tier": k.tier,
        "secret": secret,   # only returned at creation
    }


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: int,
    conn: sqlite3.Connection = Depends(get_conn),
):
    if _ak_store.get_by_id(conn, key_id) is None:
        raise HTTPException(404, f"no key with id {key_id}")
    _ak_store.revoke(conn, key_id)
```

- [ ] **Step 2: CLI `src/serve_engine/cli/key_cmd.py`**

```python
from __future__ import annotations

import asyncio
import json

import typer

from serve_engine import config
from serve_engine.cli import app, ipc

key_app = typer.Typer(help="API key management")
app.add_typer(key_app, name="key")


@key_app.command("create")
def create(
    name: str = typer.Argument(..., help="Human-readable label"),
    tier: str = typer.Option("standard", "--tier"),
):
    """Create a new API key. The secret is only printed once."""
    body = {"name": name, "tier": tier}
    result = asyncio.run(ipc.post(config.SOCK_PATH, "/admin/keys", json=body))
    typer.echo(f"id:     {result['id']}")
    typer.echo(f"name:   {result['name']}")
    typer.echo(f"tier:   {result['tier']}")
    typer.echo(f"secret: {result['secret']}")
    typer.echo("(save this secret now — it won't be shown again)")


@key_app.command("list")
def list_keys(json_out: bool = typer.Option(False, "--json")):
    """List API keys (prefixes only — secrets are never shown)."""
    keys = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/keys"))
    if json_out:
        typer.echo(json.dumps(keys, indent=2))
        return
    if not keys:
        typer.echo("no keys (auth bypassed)")
        return
    typer.echo(f"{'ID':<4} {'NAME':<20} {'TIER':<10} {'PREFIX':<14} {'REVOKED':<8}")
    for k in keys:
        revoked = "yes" if k.get("revoked") else "-"
        typer.echo(
            f"{k['id']:<4} {k['name']:<20} {k['tier']:<10} "
            f"{k['prefix']:<14} {revoked:<8}"
        )


@key_app.command("revoke")
def revoke(key_id: int = typer.Argument(...)):
    """Revoke an API key by id."""
    asyncio.run(ipc.delete(config.SOCK_PATH, f"/admin/keys/{key_id}"))
    typer.echo(f"revoked key #{key_id}")
```

- [ ] **Step 3: Register in `cli/__init__.py`**

Add `key_cmd` to the existing import block (sorted alphabetically):

```python
from serve_engine.cli import (  # noqa: F401,E402
    daemon_cmd,
    key_cmd,
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

- [ ] **Step 4: Admin tests**

Append to `tests/unit/test_admin_endpoints.py`:

```python
@pytest.mark.asyncio
async def test_create_list_revoke_key(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.post("/admin/keys", json={"name": "alice", "tier": "standard"})
        assert r.status_code == 201
        body = r.json()
        assert body["secret"].startswith("sk-")
        kid = body["id"]

        r = await c.get("/admin/keys")
        assert r.status_code == 200
        names = [k["name"] for k in r.json()]
        assert "alice" in names

        r = await c.delete(f"/admin/keys/{kid}")
        assert r.status_code == 204

        r = await c.get("/admin/keys")
        revoked = [k for k in r.json() if k["id"] == kid]
        assert revoked[0]["revoked"] is True
```

- [ ] **Step 5: Run + commit**

```bash
python -c "from serve_engine.cli import key_cmd; print('ok')"
pytest -v
ruff check src/ tests/
git add src/serve_engine/daemon/admin.py src/serve_engine/cli/key_cmd.py src/serve_engine/cli/__init__.py tests/unit/test_admin_endpoints.py
git commit -m "feat(auth): admin keys CRUD + CLI key create/list/revoke"
```

---

## Task 9: Live smoke for auth + rate limits

**Files:**
- Create: `scripts/smoke_p04_auth.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Plan 04 smoke: create a key, hit /v1/ with and without it, observe
# rate limits.

cleanup() {
    serve stop 2>/dev/null || true
    serve daemon stop 2>/dev/null || true
}
trap cleanup EXIT

serve daemon start
serve pull Qwen/Qwen2.5-0.5B-Instruct --name qwen-0_5b
serve run qwen-0_5b --gpu 0 --ctx 4096

# Phase 1: no keys yet → auth bypassed
echo "=== phase 1: no keys, expect 200 ==="
curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-0_5b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'

# Phase 2: create a trial-tier key (RPM=10)
echo "=== phase 2: create trial key ==="
secret=$(serve key create alice --tier trial | awk '/^secret:/ {print $2}')
test -n "$secret" || { echo "no secret returned"; exit 1; }
echo "Got secret: ${secret:0:12}..."

# Phase 3: hit without bearer → expect 401
echo "=== phase 3: no bearer, expect 401 ==="
curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-0_5b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'

# Phase 4: hit with good bearer → expect 200
echo "=== phase 4: good bearer, expect 200 ==="
curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST http://127.0.0.1:11500/v1/chat/completions \
  -H "Authorization: Bearer $secret" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-0_5b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'

# Phase 5: fire 15 requests rapidly → trial tier RPM=10, expect ≥ 1 429
echo "=== phase 5: 15 rapid requests, expect at least one 429 ==="
hits_429=0
for i in $(seq 1 15); do
    code=$(curl -sS -o /dev/null -w "%{http_code}" \
      -X POST http://127.0.0.1:11500/v1/chat/completions \
      -H "Authorization: Bearer $secret" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen-0_5b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')
    echo "  req $i → $code"
    if [ "$code" = "429" ]; then hits_429=$((hits_429+1)); fi
done
test $hits_429 -ge 1 || { echo "FAIL: expected at least one 429"; exit 1; }

echo "PASS ($hits_429 / 15 throttled)"
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/smoke_p04_auth.sh
git add scripts/smoke_p04_auth.sh
git commit -m "test: Plan 04 smoke (auth + rate-limit 429)"
```

---

## Verification (end of Plan 04)

1. `pytest -v` — all tests pass.
2. `ruff check src/ tests/` clean.
3. `bash scripts/smoke_p04_auth.sh` exits 0 with `PASS (N / 15 throttled)`.

## Self-review

- **Spec coverage:** API keys hashed + storable (T2), tier presets in YAML (T4), 8-window sliding limiter (T5), Bearer auth + 401 + 429+Retry-After (T6), key usage tracked from upstream responses (T7), admin CRUD + CLI (T8), end-to-end smoke (T9).
- **Backward compat:** Empty `api_keys` table → auth bypassed. Homelab users don't see a behavior change until they `serve key create` for the first time.
- **No placeholders.**
- **Type consistency:** `ApiKey`, `Limits`, `Overrides`, `Allowed`/`Denied` used identically across tasks. `tier_cfg: dict[str, Limits]` passed via `app.state` everywhere.
- **What's deferred for later:** distributed rate limiting (multi-daemon), token-bucket optimization, OAuth/OIDC integration, fine-grained per-model permissions, audit log surface beyond stdout.
