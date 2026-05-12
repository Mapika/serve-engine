-- Sub-project B (v2): snapshot index for fast warm-restore loads.
-- The snapshot blob lives at `local_path` on disk; this row is the
-- index. Federation-ready columns (source_peer_id, updated_at)
-- populated NULL/local-time until Sub-project D wires sync.
--
-- The `key` column is the content-addressable hash described in
-- docs/superpowers/specs/2026-05-13-snapshot-system-design.md §3.
-- Two deployments with the same key share a snapshot.

CREATE TABLE IF NOT EXISTS snapshots (
    id                  INTEGER PRIMARY KEY,
    key                 TEXT NOT NULL UNIQUE,
    hf_repo             TEXT NOT NULL,
    revision            TEXT NOT NULL,
    engine              TEXT NOT NULL,
    engine_image        TEXT NOT NULL,
    gpu_arch            TEXT NOT NULL,
    quantization        TEXT,
    max_model_len       INTEGER NOT NULL,
    dtype               TEXT NOT NULL,
    tensor_parallel     INTEGER NOT NULL,
    target_concurrency  INTEGER NOT NULL,
    local_path          TEXT NOT NULL,
    size_mb             INTEGER NOT NULL,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Federation-ready (Sub-project D will populate):
    source_peer_id      TEXT,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snapshots_last_used_at ON snapshots(last_used_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_hf_repo ON snapshots(hf_repo);
CREATE INDEX IF NOT EXISTS idx_snapshots_engine ON snapshots(engine);
