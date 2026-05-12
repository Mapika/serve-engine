"""Snapshot store + key computation tests (Sub-project B foundation).

Covers the schema migration + CRUD + LRU helpers + key determinism.
Engine-specific load/save methods are NOT in scope here — they land
when we wire backend.snapshot_load_argv / save_snapshot.
"""
import pytest

from serve_engine.store import db
from serve_engine.store import snapshots as snap_store
from serve_engine.store.snapshots import compute_snapshot_key


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def _add(conn, **overrides):
    base = dict(
        key="k0",
        hf_repo="Qwen/Qwen3-7B",
        revision="main",
        engine="vllm",
        engine_image="vllm/vllm-openai:v0.20.2",
        gpu_arch="9.0",
        quantization=None,
        max_model_len=4096,
        dtype="bf16",
        tensor_parallel=1,
        target_concurrency=8,
        local_path="/cache/snapshots/k0",
        size_mb=1500,
    )
    base.update(overrides)
    return snap_store.add(conn, **base)


# ---- key computation ----

def test_compute_snapshot_key_is_deterministic():
    """Same inputs → same key, every time, anywhere."""
    args = dict(
        hf_repo="Qwen/Qwen3-7B", revision="main", engine="vllm",
        engine_image="vllm:v0.20.2", gpu_arch="9.0", quantization=None,
        max_model_len=4096, dtype="bf16", tensor_parallel=1,
        target_concurrency=8,
    )
    a = compute_snapshot_key(**args)
    b = compute_snapshot_key(**args)
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_compute_snapshot_key_changes_on_any_input_change():
    """Each field is part of the key; flipping any one changes the hash."""
    base = dict(
        hf_repo="Qwen/Qwen3-7B", revision="main", engine="vllm",
        engine_image="vllm:v0.20.2", gpu_arch="9.0", quantization=None,
        max_model_len=4096, dtype="bf16", tensor_parallel=1,
        target_concurrency=8,
    )
    base_key = compute_snapshot_key(**base)
    # Each field, perturbed individually, must produce a different key.
    for field, alt in [
        ("hf_repo", "Qwen/Qwen3-1B"),
        ("revision", "v1"),
        ("engine", "sglang"),
        ("engine_image", "vllm:v0.20.3"),
        ("gpu_arch", "12.0"),
        ("quantization", "fp8"),
        ("max_model_len", 8192),
        ("dtype", "fp16"),
        ("tensor_parallel", 2),
        ("target_concurrency", 16),
    ]:
        modified = dict(base, **{field: alt})
        assert compute_snapshot_key(**modified) != base_key, (
            f"changing {field}={alt!r} did not change the key"
        )


def test_compute_snapshot_key_quantization_none_vs_str():
    """`quantization=None` and `quantization='none'` resolve to the same
    canonical input; key for `None` MUST equal the key for the literal
    string 'none' to avoid creating two snapshots for what's the same
    state. This is the design's edge-case clause from §3."""
    args_none = dict(
        hf_repo="x", revision="main", engine="vllm", engine_image="vllm:1",
        gpu_arch="9.0", quantization=None, max_model_len=1, dtype="bf16",
        tensor_parallel=1, target_concurrency=1,
    )
    args_str = dict(args_none, quantization="none")
    assert compute_snapshot_key(**args_none) == compute_snapshot_key(**args_str)


# ---- store CRUD ----

def test_add_and_get_by_key(tmp_path):
    conn = _fresh(tmp_path)
    s = _add(conn, key="abc")
    assert s.id > 0
    fetched = snap_store.get_by_key(conn, "abc")
    assert fetched is not None
    assert fetched.key == "abc"
    assert fetched.hf_repo == "Qwen/Qwen3-7B"
    assert fetched.size_mb == 1500
    assert fetched.source_peer_id is None  # locally created


def test_add_duplicate_key_raises(tmp_path):
    import sqlite3 as _sq
    conn = _fresh(tmp_path)
    _add(conn, key="dupe")
    with pytest.raises(_sq.IntegrityError):
        _add(conn, key="dupe")


def test_get_by_key_missing_returns_none(tmp_path):
    conn = _fresh(tmp_path)
    assert snap_store.get_by_key(conn, "nope") is None


def test_list_all_orders_by_last_used_desc(tmp_path):
    import time
    conn = _fresh(tmp_path)
    s1 = _add(conn, key="k1")
    time.sleep(1.1)
    s2 = _add(conn, key="k2")
    time.sleep(1.1)
    snap_store.touch(conn, s1.id)
    listed = snap_store.list_all(conn)
    # s1 was just touched → most recent → first
    assert [s.key for s in listed] == [s1.key, s2.key]


def test_list_for_engine_filters(tmp_path):
    conn = _fresh(tmp_path)
    _add(conn, key="v1", engine="vllm")
    _add(conn, key="s1", engine="sglang")
    _add(conn, key="v2", engine="vllm")
    vllm = snap_store.list_for_engine(conn, "vllm")
    assert {s.key for s in vllm} == {"v1", "v2"}


def test_touch_bumps_last_used_at(tmp_path):
    import time
    conn = _fresh(tmp_path)
    s = _add(conn, key="x")
    first = s.last_used_at
    time.sleep(1.1)
    snap_store.touch(conn, s.id)
    refreshed = snap_store.get_by_id(conn, s.id)
    assert refreshed.last_used_at > first


def test_delete_removes_row(tmp_path):
    conn = _fresh(tmp_path)
    s = _add(conn, key="rm")
    snap_store.delete(conn, s.id)
    assert snap_store.get_by_key(conn, "rm") is None


def test_total_size_mb(tmp_path):
    conn = _fresh(tmp_path)
    assert snap_store.total_size_mb(conn) == 0
    _add(conn, key="a", size_mb=1000)
    _add(conn, key="b", size_mb=500)
    _add(conn, key="c", size_mb=250)
    assert snap_store.total_size_mb(conn) == 1750


def test_lru_for_engine_model_returns_eviction_candidates(tmp_path):
    """LRU policy: keep latest N per (engine, hf_repo); rest are
    eviction candidates."""
    import time
    conn = _fresh(tmp_path)
    # Three snapshots for the same engine+repo
    s_old = _add(conn, key="old", engine="vllm", hf_repo="Qwen/Qwen3-7B")
    time.sleep(1.1)
    s_mid = _add(conn, key="mid", engine="vllm", hf_repo="Qwen/Qwen3-7B")
    time.sleep(1.1)
    s_new = _add(conn, key="new", engine="vllm", hf_repo="Qwen/Qwen3-7B")
    # And one for a different repo (must not appear in our query)
    _add(conn, key="other", engine="vllm", hf_repo="meta-llama/L8B")

    # keep_n=1 → mid + old are candidates; new is kept
    candidates = snap_store.lru_for_engine_model(
        conn, "vllm", "Qwen/Qwen3-7B", keep_n=1,
    )
    assert {c.key for c in candidates} == {s_mid.key, s_old.key}
    # keep_n=3 → no candidates (we only have 3 rows for this repo)
    assert snap_store.lru_for_engine_model(
        conn, "vllm", "Qwen/Qwen3-7B", keep_n=3,
    ) == []
    # Sanity: s_new is the survivor
    assert s_new.key == "new"


def test_migration_creates_indexes(tmp_path):
    """Indexes from 005_snapshots.sql must be in place — the predictor
    and GC paths rely on them for performance at scale."""
    conn = _fresh(tmp_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND tbl_name='snapshots'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "idx_snapshots_last_used_at" in names
    assert "idx_snapshots_hf_repo" in names
    assert "idx_snapshots_engine" in names
