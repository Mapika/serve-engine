# Sub-project B — Snapshot-based Fast Loads: Design

**Status:** Draft, ready for review (written ahead of implementation
overnight 2026-05-12 → 2026-05-13)
**Date:** 2026-05-13
**Branch:** `feat/v2-loading` (lands after Sub-project A)
**Prerequisites:** Sub-project A (adapter lifecycle) complete and stable
**Companion docs:** `2026-05-13-v2-narrative.md`,
`2026-05-13-adapter-lifecycle-design.md`

## 1. Goal

Cut cold-load time from 30-120s to <10s for repeat loads of the same
model + deployment shape. Operators stop / start / swap models throughout
the day; the cold-load wall is the single biggest UX pain point in v1.
Snapshots make warm restore the common case.

## 2. Non-goals

- **Not:** snapshot portability across GPU architectures. A snapshot
  built for sm_90 (H100) is not restored on sm_120 (Blackwell). The
  snapshot key includes gpu_arch precisely to prevent this mistake.
- **Not:** snapshot portability across engine versions. Bumping the
  vLLM image tag invalidates all vLLM snapshots. Same for SGLang,
  TRT-LLM. Old snapshots GC away.
- **Not:** federation pull-on-demand here. Schema is federation-ready
  (snapshots advertise via the same gossip primitive as adapters), but
  the cross-box pull/push lands in Sub-project D.
- **Not:** quantization-on-the-fly via snapshots. A snapshot reflects
  the engine's loaded state for one specific quant; we don't transform
  between quants.

## 3. Snapshot key

```python
snapshot_key = sha256(
    f"{hf_repo}|{revision}|{engine_name}|{engine_image_tag}"
    f"|{gpu_arch}|{quantization or 'none'}|{max_model_len}"
    f"|{dtype}|{tensor_parallel}|{target_concurrency}"
)
```

Two deployments with the same key share a snapshot. The key is
deliberately conservative — small differences (e.g., changing
`max_model_len` from 4096 to 8192) invalidate the snapshot because
engine state encodes the KV-cache layout.

`gpu_arch` is read from `nvidia-smi --query-gpu=compute_cap` per GPU
at deployment time; for multi-GPU TP deployments we require all GPUs
to have the same arch (already enforced by v1 placement).

## 4. Schema additions

```sql
-- Sub-project B (v2): snapshot index.
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY,
    key             TEXT NOT NULL UNIQUE,           -- the snapshot_key hash
    hf_repo         TEXT NOT NULL,
    revision        TEXT NOT NULL,
    engine          TEXT NOT NULL,
    engine_image    TEXT NOT NULL,                  -- the full image:tag
    gpu_arch        TEXT NOT NULL,                  -- e.g. "9.0", "12.0"
    quantization    TEXT,
    max_model_len   INTEGER NOT NULL,
    dtype           TEXT NOT NULL,
    tensor_parallel INTEGER NOT NULL,
    target_concurrency INTEGER NOT NULL,
    local_path      TEXT NOT NULL,                  -- ~/.serve/snapshots/<key>/
    size_mb         INTEGER NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Federation-ready (Sub-project D will populate):
    source_peer_id  TEXT,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_snapshots_last_used_at ON snapshots(last_used_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_hf_repo ON snapshots(hf_repo);
```

## 5. Engine-by-engine support matrix

This is where the engineering risk lives. Each engine exposes very
different surface area; the snapshot system must accommodate all three
without making the data flow leak engine-specific assumptions outward.

### vLLM (PyTorch backend, current default)
- `--load-format` accepts a `cached` mode in newer versions; the
  `torch.compile` cache lives in `~/.cache/torch/inductor/` by default.
- vLLM v0.6+ has `--enable-fast-load` (verify exact flag name) that
  uses safetensors fast-mmap.
- Realistic v1: leverage torch.compile cache + tensor mmap rather than
  a full state-dump snapshot. Reduces cold load from 60s to ~15s on
  qwen-0.5B class models. NOT a 5x improvement on first restore but
  meaningful, and zero engine-internals risk.
- Storage: bind-mount `~/.serve/snapshots/<key>/torch_cache` into the
  container at `/root/.cache/torch/`.

### SGLang (v0.5.x)
- SGLang has experimental `--cuda-graph-cache-path` for caching
  cuda graphs across runs. Plus their RadixAttention prefix cache can
  persist between sessions if pointed at a stable dir.
- Realistic v1: use cuda-graph cache + tensor mmap. Similar shape to
  vLLM's torch.compile cache.

### TRT-LLM (PyTorch backend)
- The legacy AOT path (trtllm-build) IS the snapshot — once you build
  an engine, restore is fast. But that path is deprecated by NVIDIA
  (LEGACY WARNING from session 2026-05-12).
- The PyTorch backend has limited snapshot story today. Defer
  meaningful TRT-LLM snapshot support; mark `supports_snapshots=False`
  on the backend, fall back to cold-load with a clear log line
  ("snapshot support unavailable for this engine").

**Decision:** v2.0-beta ships snapshot support for vLLM and SGLang
only. TRT-LLM gets a `supports_snapshots = False` ClassVar (mirrors
the `supports_adapters` pattern from Sub-project A). Reassess once
NVIDIA's PyTorch-backend snapshot story matures.

## 6. Lifecycle integration

In `LifecycleManager.load(plan)`, after the existing model-download +
VRAM-estimate phases, add:

```python
# 3.5: Snapshot lookup
key = compute_snapshot_key(plan, gpu_arch=topology.gpus[gpu_ids[0]].compute_cap)
snap = snapshot_store.get_by_key(conn, key)
if snap is not None and backend.supports_snapshots:
    # Mount the snapshot dir into the container; pass --load-format cached
    # (or engine-equivalent) so the engine warm-restores.
    extra_volumes = backend.snapshot_mount(snap.local_path)
    extra_argv = backend.snapshot_load_argv(snap.local_path)
else:
    extra_volumes = {}
    extra_argv = []
```

After the engine becomes healthy (existing `wait_healthy` check), kick
off a background `snapshot_save` task IF `snap is None and
backend.supports_snapshots`. Save runs ~10-30s post-load and is
non-blocking; if it fails the deployment is still fine.

`snapshot_save` calls a new backend method `save_snapshot(dep, dest_dir)`
that the backend implements per-engine (e.g., copy the relevant cache
dirs out of the container).

## 7. Backend additions

```python
class ContainerBackend:
    supports_snapshots: ClassVar[bool] = False  # default-off

    def snapshot_mount(self, snapshot_path: str) -> dict[str, dict]:
        """Volumes to add when restoring from this snapshot. Default: none."""
        return {}

    def snapshot_load_argv(self, snapshot_path: str) -> list[str]:
        """Engine flags to add when restoring (e.g., --load-format cached)."""
        return []

    async def save_snapshot(
        self, deployment: Deployment, dest_dir: Path,
    ) -> None:
        """Copy the engine's relevant cache state out to dest_dir.

        Called in the background ~10s after the engine becomes healthy.
        Implementations docker-cp the right cache subtree out of the
        container into dest_dir. May raise; caller catches and logs.
        """
        raise NotImplementedError
```

VLLMBackend overrides `supports_snapshots = True` and implements the
three methods. SGLangBackend same. TRTLLMBackend keeps the defaults
(no-op).

## 8. CLI surface (operator-facing)

```
serve snapshot ls
   # NAME (model+engine), KEY (first 8 chars), SIZE_MB, AGE, LAST_USED

serve snapshot rm <key|all>
   # Remove a snapshot from disk + index.

serve snapshot gc [--keep-last <N>] [--max-disk-gb <X>]
   # Eviction: keep N most-recently-used per (engine, model), or cap
   # total disk at X GB (LRU within the cap). Defaults configurable.
```

No `serve snapshot save` — saves are automatic post-load. Operators
who want to force a fresh save can `serve stop && serve run`.

## 9. Snapshot eviction (the GC story)

Snapshots accumulate. Without GC, `~/.serve/snapshots/` grows
unbounded. Defaults:

- **Keep latest N per (engine, model):** N=2. So one snapshot per
  recent config; one fallback for the previous shape.
- **Total disk cap:** none by default. Operators set `disk_quota_gb`
  in `~/.serve/snapshots.yaml` (new config file); when exceeded, LRU
  evicts.
- **Run GC opportunistically:** at daemon startup, after any save, and
  via cron-like background timer (every 6h).

## 10. Federation hooks (Sub-project D will use)

Schema columns ready:
- `snapshots.source_peer_id` — NULL = locally created.
- `snapshots.updated_at` — last-write-wins reconciliation.

When D lands:
- Push: snapshot index entries (NOT the blob) gossip to peers on save.
- Pull: when local lookup misses but a peer has a matching key, HTTP
  GET the blob over to local disk before invoking the engine restore.
- Conflict: same-key snapshots are byte-identical (key is content-
  addressable per its definition); no real conflict possible. Choose
  the closest peer for pull.

## 11. Testing strategy

- `test_snapshot_key.py` — key determinism: same plan inputs → same
  key; any input changes → different key. Snapshot key must NOT
  include any non-deterministic fields (timestamps, peer IDs).
- `test_snapshot_store.py` — CRUD; LRU eviction within quota.
- `test_vllm_backend_snapshot.py` — `snapshot_load_argv` shape;
  `snapshot_mount` shape.
- `test_sglang_backend_snapshot.py` — same for SGLang.
- `test_lifecycle_snapshot_integration.py` — end-to-end: deploy a
  model → snapshot saves → stop → re-deploy with same plan → snapshot
  restore path is taken. Uses mocked engine HTTP layer; verifies the
  argv contains the snapshot-load flag.

Live verification (operator):
- Deploy qwen3-0_6b on vLLM, time the load.
- Stop, re-deploy with the same flags, time the second load.
- Expect ≥3x speedup on the second load. (5x is the stretch goal.)

## 12. Decisions I'm flagging for review

- **Snapshot save timing.** Currently proposing 10s post-healthy.
  Alternative: save on graceful shutdown only (avoids saving an
  unstable engine state). Trade-off: shutdown saves don't help if
  daemon crashes. Lean toward post-healthy for correctness.
- **Save failures.** If the background snapshot_save task fails, what
  happens? Proposal: log + emit lifecycle event + retry once after 30s.
  Don't fail the deployment; the engine is still serving fine.
- **Snapshot layout on disk.** Proposal: `~/.serve/snapshots/<key>/`
  contains a `manifest.json` (engine, version, original plan inputs,
  size) plus one or more cache subdirs (`torch_cache/`, `cuda_graphs/`,
  etc.). Operators can `du -sh ~/.serve/snapshots/*` to see costs.
- **vLLM and SGLang exact flag names + cache-dir paths** — verify
  against pinned versions before locking. Each engine's docs are
  unstable here; prefer reading their actual CLI `--help` output in
  containers.
- **CONFIGS_DIR / SNAPSHOTS_DIR layout in `~/.serve/`.** Sub-project A
  added `~/.serve/configs/`. Sub-project B adds `~/.serve/snapshots/`.
  At some point the directory tree should be documented in README.

## 13. What's intentionally NOT in scope

- TRT-LLM snapshot support (deferred until NVIDIA's PyTorch-backend
  snapshot story matures)
- Cross-architecture snapshot translation (keys explicitly include
  gpu_arch to prevent corruption)
- Snapshot diff/incremental updates (full-copy is simpler; snapshots
  are already content-addressable)
- Snapshot encryption-at-rest (the operator's filesystem is trusted;
  no multi-tenant story)
- Web UI for snapshots (lands with the v2 UI deepening work)
