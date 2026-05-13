"""Snapshot GC: the two-step eviction policy + the YAML config loader.

The background loop itself just calls tick_once on a timer, so the
testing focus is run_gc semantics and the snapshot config-file roundtrip.
"""
from __future__ import annotations

from pathlib import Path

from serve_engine.lifecycle.snapshot_gc import SnapshotGcConfig, run_gc
from serve_engine.store import db
from serve_engine.store import snapshots as snap_store


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def _seed_snapshots(conn, snap_root: Path, count: int, *, engine: str, hf_repo: str):
    """Seed `count` snapshots with predictable keys + last_used_at stagger.
    Returns the rows newest-to-oldest."""
    out = []
    for i in range(count):
        d = snap_root / f"{engine}-{i}"
        d.mkdir(parents=True)
        (d / "torch_cache").mkdir()
        (d / "torch_cache" / "blob.bin").write_bytes(b"x" * 1024)
        s = snap_store.add(
            conn, key=f"{engine}-key-{i}",
            hf_repo=hf_repo, revision="main",
            engine=engine, engine_image=f"{engine}:img",
            gpu_arch="9.0", quantization=None,
            max_model_len=4096, dtype="auto",
            tensor_parallel=1, target_concurrency=8,
            local_path=str(d), size_mb=1,
        )
        conn.execute(
            "UPDATE snapshots SET last_used_at = datetime('now', ?) WHERE id=?",
            (f"-{i * 10} minutes", s.id),
        )
        out.append(s)
    return out


def test_run_gc_keeps_n_per_engine_model(tmp_path):
    """keep_last_per_model=2: with 4 snapshots of the same engine+model,
    the 2 oldest are evicted (rows + on-disk dirs gone)."""
    conn = _fresh(tmp_path)
    snap_root = tmp_path / "snaps"
    rows = _seed_snapshots(conn, snap_root, 4, engine="vllm", hf_repo="org/m")

    result = run_gc(conn, keep_last_per_model=2, max_disk_gb=None)

    assert result["removed"] == 2
    remaining = snap_store.list_all(conn)
    assert len(remaining) == 2
    # Newest two survive (last_used_at staggered by 10 min, lower i = newer).
    surviving_keys = {r.key for r in remaining}
    assert surviving_keys == {rows[0].key, rows[1].key}
    # The two evicted dirs are gone.
    assert not (snap_root / "vllm-2").exists()
    assert not (snap_root / "vllm-3").exists()


def test_run_gc_independent_per_engine_model_pair(tmp_path):
    """Per-pair eviction: vLLM/m and SGLang/m each keep their own N."""
    conn = _fresh(tmp_path)
    snap_root = tmp_path / "snaps"
    _seed_snapshots(conn, snap_root, 3, engine="vllm", hf_repo="org/m")
    _seed_snapshots(conn, snap_root, 3, engine="sglang", hf_repo="org/m")

    result = run_gc(conn, keep_last_per_model=1, max_disk_gb=None)

    assert result["removed"] == 4  # 2 per engine
    rows = snap_store.list_all(conn)
    by_engine: dict[str, int] = {}
    for r in rows:
        by_engine[r.engine] = by_engine.get(r.engine, 0) + 1
    assert by_engine == {"vllm": 1, "sglang": 1}


def test_run_gc_max_disk_gb_global_lru(tmp_path):
    """When total size exceeds max_disk_gb, oldest snapshots evict
    globally until under cap — even crossing engine/model boundaries."""
    conn = _fresh(tmp_path)
    snap_root = tmp_path / "snaps"
    # 5 snapshots, each 1 MB, last_used_at staggered.
    _seed_snapshots(conn, snap_root, 5, engine="vllm", hf_repo="org/m")
    # Cap at 0.003 GB = 3 MB → must keep at most 3 rows.
    result = run_gc(conn, keep_last_per_model=0, max_disk_gb=0.003)
    assert result["removed"] == 2
    assert snap_store.total_size_mb(conn) == 3
    assert result["remaining_mb"] == 3


def test_run_gc_keep_last_zero_disables_per_model(tmp_path):
    """keep_last_per_model=0 is a documented escape: skip step 1, rely
    only on the global disk cap."""
    conn = _fresh(tmp_path)
    snap_root = tmp_path / "snaps"
    _seed_snapshots(conn, snap_root, 4, engine="vllm", hf_repo="org/m")
    result = run_gc(conn, keep_last_per_model=0, max_disk_gb=None)
    assert result["removed"] == 0
    assert len(snap_store.list_all(conn)) == 4


def test_run_gc_empty_db_is_noop(tmp_path):
    conn = _fresh(tmp_path)
    result = run_gc(conn, keep_last_per_model=2, max_disk_gb=1.0)
    assert result == {"removed": 0, "remaining_mb": 0}


def test_config_load_returns_defaults_when_file_absent(tmp_path):
    cfg = SnapshotGcConfig.load(tmp_path / "missing.yaml")
    assert cfg.keep_last_per_model == 2
    assert cfg.max_disk_gb is None
    assert cfg.tick_s == 6 * 3600.0


def test_config_load_reads_yaml_overrides(tmp_path):
    p = tmp_path / "snapshots.yaml"
    p.write_text("keep_last_per_model: 5\nmax_disk_gb: 100\ntick_s: 120\n")
    cfg = SnapshotGcConfig.load(p)
    assert cfg.keep_last_per_model == 5
    assert cfg.max_disk_gb == 100.0
    assert cfg.tick_s == 120.0


def test_config_load_malformed_yaml_falls_back_to_defaults(tmp_path):
    p = tmp_path / "snapshots.yaml"
    p.write_text("not: valid: yaml:")  # parse error
    cfg = SnapshotGcConfig.load(p)
    assert cfg == SnapshotGcConfig()
