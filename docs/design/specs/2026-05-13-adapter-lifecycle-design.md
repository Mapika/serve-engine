# Sub-project A — Adapter-First Lifecycle: Design

**Status:** Draft, ready for review
**Date:** 2026-05-13
**Branch:** `feat/v2-loading`
**Sequence:** First sub-project of v2 (precedes B Snapshots, C Predictive, D Federation)
**Companion docs:** `2026-05-13-v2-narrative.md`

## 1. Goal

Make LoRA / DoRA adapters first-class entities alongside base models.
Operators register adapters; clients address them by name in OpenAI
requests; the daemon routes the request to a deployment that has the
right base + adapter loaded; adapters hot-swap per request in
milliseconds — orders of magnitude cheaper than full model swaps.

This is the highest-leverage v2 feature for single-box users. Today,
"swapping models" means a 30-120s container restart. Tomorrow,
"swapping adapters" means routing a single request differently against
a base that's already warm.

## 2. Non-goals

- **Not:** training, fine-tuning, or generating adapters. Operator
  brings the adapter (HF repo or local file).
- **Not:** TRT-LLM adapter support. TRT-LLM has a thinner adapter story
  via the legacy AOT engine path; the PyTorch backend's LoRA support
  is in flux. Returns "adapter unsupported on this engine" with a
  clear error.
- **Not:** cross-engine adapter portability. Adapter trained for vLLM
  loading is loaded by vLLM. We don't translate formats.
- **Not:** federation in this sub-project. Schema is designed
  federation-ready; sync is implemented in Sub-project D.
- **Not:** adapter composition (multiple adapters merged per request).
  Engines don't reliably support this; out of scope.

## 3. Schema additions

```sql
-- New table for adapters. Adapters are tied to a base model by name.
CREATE TABLE adapters (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,         -- the addressable name
    base_model_id INTEGER NOT NULL REFERENCES models(id),
    hf_repo      TEXT NOT NULL,                -- HF repo or local path
    revision     TEXT NOT NULL DEFAULT 'main',
    local_path   TEXT,                         -- populated after download
    size_mb      INTEGER,                      -- populated after download
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Federation-ready (Sub-project D will populate):
    source_peer_id TEXT,                       -- NULL = locally registered
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_adapters_base_model_id ON adapters(base_model_id);

-- Junction table: which adapters are loaded into which deployment.
-- A row appears when an adapter is hot-loaded into a running deployment.
CREATE TABLE deployment_adapters (
    deployment_id INTEGER NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    adapter_id   INTEGER NOT NULL REFERENCES adapters(id),
    loaded_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (deployment_id, adapter_id)
);
```

The `models` table gets one new column to mark which models are
adapter-capable (i.e., which backends are loaded with LoRA enabled):

```sql
ALTER TABLE deployments ADD COLUMN max_loras INTEGER DEFAULT 0;
-- 0 = LoRA disabled; >0 = engine started with --enable-lora --max-loras N
```

Migration: idempotent additive — `CREATE TABLE IF NOT EXISTS`,
`ALTER TABLE … ADD COLUMN` guarded by checking `pragma_table_info`.

## 4. CLI surface

```
serve adapter pull <hf-repo> --base <base-model-name> --name <adapter-name>
   # Register + download. --name optional (defaults to repo basename).

serve adapter ls
   # List registered adapters with their base, size, last-used.

serve adapter rm <name>
   # Remove the adapter from the registry. Refuses if loaded into
   # any active deployment unless --force.

serve adapter load <name> [--deployment <dep-id>]
   # Hot-load the adapter into a ready deployment of its base.
   # If --deployment omitted, picks the most-recently-used deployment
   # of the base, or errors if none.

serve adapter unload <name> [--deployment <dep-id>]
   # Hot-unload (frees an engine adapter slot).

serve run <base-name-or-repo> --max-loras <N>
   # Start a deployment with LoRA enabled. N is the number of adapter
   # slots the engine will reserve for hot-swapping. Default: 0 (LoRA
   # disabled). Recommended: 4-8 for typical workloads.
```

Inference clients see no new CLI — they just say
`model='my-adapter-v3'` (or `model='qwen3-7b:my-adapter-v3'`) in their
OpenAI request.

## 5. HTTP API additions

```
GET    /v1/models
   # Existing endpoint, expanded. Returns base models AND adapters.
   # Adapter entries include `base: "<base-model-name>"` for clients
   # that care to disambiguate.

POST   /admin/adapters
   { "name": "...", "base_model_name": "...", "hf_repo": "...",
     "revision": "main" }
   # Register an adapter. Returns 201 + adapter row.

GET    /admin/adapters
   # List adapters with status (registered, downloaded, loaded-into).

DELETE /admin/adapters/{name}
   # Remove from registry. Adapter blob NOT auto-deleted from disk.

POST   /admin/adapters/{name}/download
   # Pull the adapter from HF / verify a local path. Idempotent.

POST   /admin/deployments/{id}/adapters/{adapter_name}
   # Hot-load adapter into a running deployment.
   # 404 if either is missing; 409 if engine doesn't support adapters
   # or all adapter slots are full.

DELETE /admin/deployments/{id}/adapters/{adapter_name}
   # Hot-unload. Returns 204.
```

OpenAI proxy (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`):
- Look up `model` field. If it matches a registered adapter name (or
  `base_name:adapter_name` form), resolve to (base_model, adapter).
- Find an active deployment of the base with the adapter loaded.
  - If found: dispatch with `model=adapter_name` in the upstream
    payload (vLLM/SGLang both interpret this as "use this adapter").
  - If not found: trigger lifecycle to either hot-load the adapter
    into an existing deployment or start a new one with LoRA enabled.
  - Update `last_used_at` for the (deployment, adapter) row.

## 6. Engine integration

### vLLM
- Static (start-time) loading: `--enable-lora --max-loras N
  --lora-modules <name1>=<path1> <name2>=<path2>`. Adapters listed at
  start are always loaded.
- Dynamic (runtime) loading: vLLM v0.6+ exposes
  `POST /v1/load_lora_adapter { lora_name, lora_path }` and
  `POST /v1/unload_lora_adapter { lora_name }`. We use this for hot
  load/unload after the deployment is up.
- Inference: client sets `model=lora_name`; vLLM matches against loaded
  adapter names.
- Constraints: `--max-loras` caps concurrent slots; `--max-lora-rank`
  caps the rank dimension we'll accept.

### SGLang
- Start-time loading: `--lora-paths <name>=<path> <name>=<path>`.
  Slot count derived from how many `--lora-paths` entries exist.
- Dynamic loading: SGLang v0.5.x has `/load_lora_adapter` and
  `/unload_lora_adapter` (similar shape to vLLM but slightly different
  payload — needs verification against the pinned tag).
- Inference: same `model=lora_name` convention.

### TRT-LLM
- Adapter support exists for the legacy AOT-engine path but is fragile
  and incompatible with the PyTorch backend we use. Out of scope for
  this sub-project. Backend's `engine_config()` does NOT advertise
  adapter support; the lifecycle returns 409 if a user tries to load
  an adapter into a TRT-LLM deployment.
- Backend gets a `supports_adapters: bool = False` class attribute.
  `ContainerBackend` defaults to True (adapter support enabled by
  default for new backends); TRTLLMBackend overrides to False.

## 7. Routing (proxy changes)

Today the proxy resolves `model` → deployment via
`dep_store.find_ready_by_model_name`. New flow:

```
def resolve_target(model_field: str) -> tuple[Deployment, str | None]:
    """Returns (deployment, adapter_name_or_None)."""
    # 1. Composite form: "base:adapter"
    if ":" in model_field:
        base_name, adapter_name = model_field.split(":", 1)
        adapter = adapter_store.get_by_name(adapter_name)
        # validate adapter.base_model.name == base_name
        ...
    # 2. Bare adapter name
    elif (a := adapter_store.get_by_name(model_field)) is not None:
        adapter = a
        base_name = a.base_model.name
    # 3. Bare base name (existing v1 path)
    else:
        adapter = None
        base_name = model_field

    dep = find_deployment_for(base_name, adapter)
    return dep, adapter.name if adapter else None
```

`find_deployment_for(base, adapter)`:
1. Prefer a ready deployment of `base` with `adapter` already loaded
   (junction table lookup).
2. Else: prefer a ready deployment of `base` with LoRA enabled
   (`max_loras > 0`) and a free adapter slot — hot-load the adapter
   into it.
3. Else: prefer a ready deployment of `base` with LoRA enabled and full
   slots — evict the LRU adapter from that deployment, hot-load this
   one.
4. Else: trigger a new deployment of `base` with LoRA enabled.

Step 4 is slow (full cold load). Steps 2 + 3 are sub-second.

The proxy passes `model=adapter_name` (not `model=base_name`) to the
upstream engine when an adapter is in play. This matches vLLM/SGLang
expectations.

## 8. Eviction

Two layers:
1. **Deployment-level eviction (existing v1 LRU):** unchanged. Bases
   evict at full-deployment granularity. Existence of loaded adapters
   does NOT prevent base eviction; if a base is evicted, its
   `deployment_adapters` rows are cascaded out.
2. **Adapter-within-deployment eviction (new):** when a deployment's
   adapter slots are full and a new adapter must load, the LRU adapter
   (by `deployment_adapters.last_used_at`) is hot-unloaded first. The
   adapter remains in the registry; it's just no longer loaded into
   that deployment.

`--pin` on a deployment still prevents base eviction. There is no
adapter-level pin in v2.0; can add later if requested.

## 9. Federation hooks (designed in, NOT implemented here)

Schema fields ready for Sub-project D:
- `adapters.source_peer_id` — NULL = locally registered, else the peer
  UUID where this adapter was first registered.
- `adapters.updated_at` — used for last-write-wins reconciliation.

When D lands:
- Push: when an adapter is added/modified locally, push to peers.
- Pull: when a request references an adapter not present locally, the
  proxy queries peers for the adapter blob and pulls it.
- Conflict: if two peers register an adapter with the same name
  independently, the one with the later `updated_at` wins.

For sub-project A: those columns are populated only with NULL /
local-time. The sync logic stays empty. The schema is correct so D
can be additive.

## 10. Testing strategy

Unit tests:
- `test_adapter_store.py` — CRUD, by-name lookup, base validation
- `test_adapter_routing.py` — `resolve_target` for all 4 routing cases
  (composite, bare adapter, bare base, missing); `find_deployment_for`
  preference ordering
- `test_adapter_eviction.py` — LRU within deployment, no spillover
  to base eviction
- `test_vllm_backend_lora.py` — argv generation when `max_loras > 0`
  (`--enable-lora`, `--max-loras N`, `--lora-modules name=path` for
  start-time-loaded adapters)
- `test_sglang_backend_lora.py` — same for SGLang
- `test_trtllm_backend_lora.py` — confirms `supports_adapters=False`
  and that registering an adapter against a TRT-LLM deployment errors
  cleanly

Integration tests:
- Mock vLLM container with a fake `/v1/load_lora_adapter` endpoint;
  verify the lifecycle drives it correctly
- End-to-end via `serve adapter pull` → `serve run` → curl
  `/v1/chat/completions` with `model=adapter_name` — uses `monkeypatch`
  on the engine HTTP layer; doesn't need a real GPU

Live verification (operator-driven, not in CI):
- Real Qwen3 base + a small LoRA adapter from HF
- Sub-second swap between adapters under load (use bench harness with
  mixed adapter requests)

## 11. Decisions I'm flagging for review

These are choices I'd make solo while implementing if not flagged.
Calling them out now so they get user input before they harden.

- **Adapter slot default.** What's a sensible default for `--max-loras`
  when not specified? Leaning toward 4. (vLLM and SGLang both have
  modest per-slot memory cost; 4 is "few enough to not waste, enough to
  matter.")
- **`serve adapter pull` vs `serve adapter add`.** v1 uses `serve pull`
  for "register + download" of base models. Mirroring as `serve adapter
  pull` is consistent. But `add` is a clearer verb when the source is
  a local file. **Proposal: `serve adapter pull <hf-repo>` AND
  `serve adapter add <local-path> --base <base>`**.
- **Adapter naming collisions with base models.** A user could
  accidentally register an adapter named identically to a base model.
  Routing has an ambiguity (#3 in `resolve_target` would never trigger
  for that name). **Proposal: enforce a disjoint-namespace check at
  registration time — adapter name must not collide with any
  `models.name`, and vice versa.**
- **The `/admin/adapters/{name}/download` retry/cancellation story.**
  Adapters are smaller than bases (10-200 MB) so this matters less,
  but the same operator-cancellation concerns apply. **Proposal: same
  as v1 `serve pull` — synchronous endpoint, client cancels by closing
  connection; partial downloads are HF's snapshot-resume story.**
- **Whether to expose `max_loras` on `serve run`.** Current proposal:
  yes, as `--max-loras N`. Alternative: implicit (auto-enable LoRA on
  any deployment of a base that has registered adapters). Implicit is
  more magical but less predictable. **Sticking with explicit.**

## 12. What's intentionally NOT in scope

- Adapter sync between peers (Sub-project D)
- Adapter discovery / search (operator brings the HF repo)
- Adapter performance benchmarking (use the v1 bench harness with mixed
  `model=` traffic — the harness already takes a list of model names)
- Adapter version/tag management (treat each `(repo, revision)` pair as
  a distinct adapter; operators name them however they want)
- UI for adapters (lands in v2 UI deepening work, not this sub-project)
