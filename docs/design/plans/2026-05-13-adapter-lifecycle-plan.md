# Workstream A - Adapter Lifecycle: Implementation Plan

**Goal:** Land adapter-as-first-class-entity on `feat/v2-loading`. After this plan, registering a LoRA adapter is one CLI call; OpenAI clients can address it as `model='<adapter-name>'`; the proxy hot-loads the adapter into a running deployment of its base in sub-second time and dispatches the request there.

**Branch:** `feat/v2-loading` (off `main`).
**Companion design:** `docs/design/specs/2026-05-13-adapter-lifecycle-design.md`.
**Subsequent work:** Workstream B (Snapshot system) and C (Predictive layer) land on the same branch in subsequent plans. Workstream D (Federation) branches off this one.

**Tech stack:** Same as v1. New: dynamic LoRA load/unload via vLLM `/v1/load_lora_adapter` and SGLang `/load_lora_adapter` (HTTP, no new Python deps).

---

## File structure

```
serving-engine/
|-- src/serve_engine/
|   |-- store/
|   |   |-- adapters.py                # NEW - adapter store (CRUD)
|   |   |-- deployment_adapters.py     # NEW - junction store + LRU
|   |   +-- migrations/
|   |       +-- 0002_adapters.sql      # NEW - schema additions
|   |-- lifecycle/
|   |   |-- adapter_downloader.py      # NEW - pull adapter from HF
|   |   |-- adapter_router.py          # NEW - resolve_target + find_deployment_for
|   |   +-- manager.py                 # MODIFIED - adapter hot-load/unload
|   |-- backends/
|   |   |-- base.py                    # MODIFIED - supports_adapters class attr
|   |   |-- trtllm.py                  # MODIFIED - supports_adapters = False
|   |   |-- vllm.py                    # MODIFIED - --enable-lora --max-loras N
|   |   +-- sglang.py                  # MODIFIED - --lora-paths
|   |-- daemon/
|   |   |-- admin.py                   # MODIFIED - /admin/adapters/* routes
|   |   +-- openai_proxy.py            # MODIFIED - adapter resolution + dispatch
|   |-- cli/
|   |   |-- adapter_cmd.py             # NEW - serve adapter pull/ls/rm/load/unload
|   |   |-- run_cmd.py                 # MODIFIED - --max-loras
|   |   +-- __init__.py                # MODIFIED - register adapter_cmd
|   +-- lifecycle/
|       +-- plan.py                    # MODIFIED - DeploymentPlan.max_loras: int = 0
+-- tests/
    +-- unit/
        |-- test_adapter_store.py
        |-- test_deployment_adapters_store.py
        |-- test_adapter_downloader.py
        |-- test_adapter_router.py
        |-- test_adapter_eviction.py
        |-- test_vllm_backend_lora.py
        |-- test_sglang_backend_lora.py
        |-- test_trtllm_backend_lora.py
        |-- test_admin_adapter_endpoints.py
        +-- test_proxy_adapter_dispatch.py
```

---

## Task 1: Schema migration

**Files:**
- Create: `src/serve_engine/store/migrations/0002_adapters.sql`

The migration is additive and idempotent so v1 -> v2 upgrade is a no-op
on first start. Schema mirrors the design doc Âsection3.

- [ ] **Step 1**: Write the SQL.
- [ ] **Step 2**: Confirm `db.init_schema` picks up the new file (sorted by name).
- [ ] **Step 3**: Smoke test - `db.connect(tmp); init_schema(); pragma_table_info('adapters')` returns expected columns.

---

## Task 2: Adapter store

**Files:**
- Create: `src/serve_engine/store/adapters.py`
- Create: `tests/unit/test_adapter_store.py`

Mirror `store/models.py` shape. Functions: `add(conn, name, base_model_name, hf_repo, revision='main')`, `get_by_name(conn, name)`, `get_by_id(conn, id)`, `list_all(conn)`, `set_local_path(conn, id, path)`, `set_size_mb(conn, id, mb)`, `delete(conn, id)`.

Dataclass `Adapter(id, name, base_model, hf_repo, revision, local_path, size_mb, created_at, source_peer_id, updated_at)`. `base_model` is a resolved `Model` object (one extra query per fetch - fine for the volumes involved).

Validation in `add`:
- Reject if `name` collides with any existing `models.name` or `adapters.name`.
- Reject if `base_model_name` doesn't resolve to a model.
- Set `updated_at = CURRENT_TIMESTAMP` (federation-ready).

- [ ] **Step 1**: Dataclass + `_row_to_adapter` helper.
- [ ] **Step 2**: `add` with collision check.
- [ ] **Step 3**: Read functions (`get_by_name`, `get_by_id`, `list_all`).
- [ ] **Step 4**: Mutation functions (`set_local_path`, `set_size_mb`, `delete`).
- [ ] **Step 5**: Tests - happy path, collision rejection (vs models, vs adapters), missing base, set_local_path roundtrip.

---

## Task 3: Deployment-adapter junction store

**Files:**
- Create: `src/serve_engine/store/deployment_adapters.py`
- Create: `tests/unit/test_deployment_adapters_store.py`

Functions:
- `attach(conn, dep_id, adapter_id)` - inserts row, idempotent on duplicate (touches `loaded_at`/`last_used_at`).
- `detach(conn, dep_id, adapter_id)` - deletes row.
- `list_for_deployment(conn, dep_id) -> list[Adapter]` - JOIN on adapters.
- `touch(conn, dep_id, adapter_id)` - updates `last_used_at`. Called by the proxy on every dispatched request.
- `lru_for_deployment(conn, dep_id) -> Adapter | None` - returns the LRU loaded adapter, for eviction.
- `find_deployments_with_adapter(conn, adapter_id) -> list[int]` - for routing.

CASCADE on deployment delete - adapter junction rows clean up automatically.

- [ ] **Step 1**: Implement functions.
- [ ] **Step 2**: Tests - attach idempotency, detach, lru ordering, cascade behavior.

---

## Task 4: Backend `supports_adapters` flag

**Files:**
- Modify: `src/serve_engine/backends/base.py`
- Modify: `src/serve_engine/backends/trtllm.py`
- Modify: `src/serve_engine/backends/vllm.py`
- Modify: `src/serve_engine/backends/sglang.py`

Add `supports_adapters: ClassVar[bool] = True` to `ContainerBackend`. Set to `False` in `TRTLLMBackend` (with a brief comment pointing at the design doc's TRT-LLM section).

vLLM and SGLang don't need code changes here; they inherit the True default.

- [ ] **Step 1**: Add the class attribute to `ContainerBackend`.
- [ ] **Step 2**: Override in `TRTLLMBackend`.
- [ ] **Step 3**: Confirm existing tests still pass.

---

## Task 5: Plan + manager wire-through for `max_loras`

**Files:**
- Modify: `src/serve_engine/lifecycle/plan.py`
- Modify: `src/serve_engine/lifecycle/manager.py`
- Modify: `src/serve_engine/daemon/admin.py` (CreateDeploymentRequest)
- Modify: `src/serve_engine/cli/run_cmd.py` (--max-loras flag)

Add `max_loras: int = 0` to `DeploymentPlan`. Validation: `max_loras >= 0`; if `> 0`, the chosen backend must have `supports_adapters = True` (raise ValueError otherwise).

Manager passes `max_loras` through to backends via the existing
`effective_plan` mechanism. Backends that respect it append the right
flags (Task 6).

- [ ] **Step 1**: Add field + validation to `DeploymentPlan`.
- [ ] **Step 2**: Plumb through `CreateDeploymentRequest` and the
  manager's `effective_plan`.
- [ ] **Step 3**: Add `--max-loras N` to `serve run`.
- [ ] **Step 4**: Test the validation rejects `max_loras > 0` with a
  TRT-LLM backend.

---

## Task 6: vLLM and SGLang argv emission

**Files:**
- Modify: `src/serve_engine/backends/vllm.py`
- Modify: `src/serve_engine/backends/sglang.py`
- Create: `tests/unit/test_vllm_backend_lora.py`
- Create: `tests/unit/test_sglang_backend_lora.py`

vLLM `build_argv`: when `plan.max_loras > 0`, append
`["--enable-lora", "--max-loras", str(plan.max_loras)]`.
At-start adapter loading via `--lora-modules name=path,name=path,...`
is NOT done here - all adapters load dynamically post-startup. Reason:
makes the deployment-creation path independent of "which adapters
exist for this base," which keeps Task 7 simpler.

SGLang `build_argv`: when `plan.max_loras > 0`, append
`["--enable-lora-async-loading"]` (verify exact flag name against
v0.5.x - flag is needed for runtime adapter ops). Slot count is
implicit from the flag.

- [ ] **Step 1**: vLLM argv emission + tests.
- [ ] **Step 2**: SGLang argv emission + tests. **Verify flag name**
  by spinning up an SGLang container with `--help` if the docs are
  unclear.
- [ ] **Step 3**: Confirm TRT-LLM doesn't change behavior.

---

## Task 7: Adapter downloader

**Files:**
- Create: `src/serve_engine/lifecycle/adapter_downloader.py`
- Create: `tests/unit/test_adapter_downloader.py`

Mirrors `lifecycle/downloader.py`. One function:
`download_adapter(*, hf_repo, revision, cache_dir) -> tuple[str, int]`.
Returns `(local_path, size_mb)`. Uses
`huggingface_hub.snapshot_download` (same as base models). Adapter
dirs are typically <500 MB, so synchronous + simple is fine.

`size_mb` calculation: walk the snapshot dir and sum file sizes;
divide by `1024 * 1024`, round up. Stored in `adapters.size_mb` for UI
display.

- [ ] **Step 1**: Implement function.
- [ ] **Step 2**: Test with mocked `snapshot_download`; assert returned
  path and computed `size_mb`.

---

## Task 8: HTTP admin endpoints

**Files:**
- Modify: `src/serve_engine/daemon/admin.py`
- Create: `tests/unit/test_admin_adapter_endpoints.py`

Add the 5 endpoints from design Âsection5:
1. `POST /admin/adapters` - register
2. `GET /admin/adapters` - list with status (registered, downloaded, loaded-into)
3. `DELETE /admin/adapters/{name}` - remove from registry
4. `POST /admin/adapters/{name}/download` - download blob
5. `POST /admin/deployments/{id}/adapters/{adapter_name}` - hot-load
6. `DELETE /admin/deployments/{id}/adapters/{adapter_name}` - hot-unload

Hot-load endpoint:
- Verify deployment is `ready` and backend `supports_adapters`.
- If adapter slots full: hot-unload the LRU adapter first.
- Call upstream engine via httpx:
  - vLLM: `POST {engine}/v1/load_lora_adapter { lora_name, lora_path }`
  - SGLang: `POST {engine}/load_lora_adapter { lora_name, lora_path }` (verify exact path)
- On success, insert `deployment_adapters` row.

Hot-unload endpoint:
- Call upstream engine: `POST {engine}/v1/unload_lora_adapter { lora_name }`
- Delete `deployment_adapters` row.

The `lora_path` is the in-container path: `/cache/<adapter_local_path>`.

- [ ] **Step 1**: Endpoints 1-4 (register / list / delete / download).
- [ ] **Step 2**: Endpoint 5 (hot-load) including LRU eviction.
- [ ] **Step 3**: Endpoint 6 (hot-unload).
- [ ] **Step 4**: Tests using mocked httpx for the upstream calls.

---

## Task 9: Proxy resolution and dispatch

**Files:**
- Create: `src/serve_engine/lifecycle/adapter_router.py`
- Modify: `src/serve_engine/daemon/openai_proxy.py`
- Create: `tests/unit/test_adapter_router.py`
- Create: `tests/unit/test_proxy_adapter_dispatch.py`

`adapter_router.resolve_target(conn, model_field) -> ResolvedTarget`
where `ResolvedTarget = (deployment_id, base_model_name,
adapter_name_or_none)`. Implements the 3 routing cases from design Âsection7.

`adapter_router.find_deployment_for(conn, manager, base, adapter)`
implements the 4-step preference ordering (preloaded > hot-loadable >
slot-evict-and-load > new-deployment).

The proxy wires this in:
- Replace `dep_store.find_ready_by_model_name(conn, body['model'])`
  with `adapter_router.resolve_target(...) -> find_deployment_for(...)`.
- When dispatching upstream, send `model=adapter_name` if an adapter
  is in play (vLLM/SGLang convention); otherwise pass `model=base_name`
  unchanged.
- Call `deployment_adapters.touch(...)` after a successful adapter
  dispatch.

- [ ] **Step 1**: Implement `resolve_target`.
- [ ] **Step 2**: Implement `find_deployment_for` with preference
  ordering tests.
- [ ] **Step 3**: Wire into `openai_proxy.py`.
- [ ] **Step 4**: Tests including mocked downstream - assert correct
  `model` value on the upstream call.

---

## Task 10: CLI ergonomics

**Files:**
- Create: `src/serve_engine/cli/adapter_cmd.py`
- Modify: `src/serve_engine/cli/__init__.py`

`serve adapter` Typer sub-app with subcommands:
- `pull <hf-repo> --base <name> [--name <adapter-name>] [--revision main]`
  - registers + downloads.
- `add <local-path> --base <name> --name <adapter-name>` - registers a
  pre-downloaded adapter (skips HF pull). Sets `local_path` directly.
- `ls` - table: NAME, BASE, SIZE_MB, LOADED_INTO (count of deployments).
- `rm <name> [--force]` - removes from registry. Refuses if loaded
  unless `--force` (which hot-unloads from all deployments first).
- `load <name> [--deployment <id>]` - hot-loads. Default deployment
  picked via "ready, supports_adapters, base matches, MRU".
- `unload <name> [--deployment <id>]` - hot-unloads.

All commands use the existing `ipc` module to talk to the daemon over
the UDS, same pattern as `pull_cmd.py` etc.

- [ ] **Step 1**: Wire the Typer sub-app + register in `__init__.py`.
- [ ] **Step 2**: Implement subcommands one at a time (pull -> ls -> load
  -> unload -> rm -> add).
- [ ] **Step 3**: Run `serve adapter --help` and verify the surface.

---

## Task 11: End-to-end test

**Files:**
- Create: `tests/integration/test_adapter_lifecycle.py`

Full HTTP flow with all engine I/O mocked:
1. Start a deployment of base `qwen3-test` with `--max-loras 4`.
2. Register adapter `tone-formal` against `qwen3-test`.
3. Hot-load the adapter into the deployment.
4. POST `/v1/chat/completions` with `model='tone-formal'`. Assert the
   mocked upstream sees `model='tone-formal'` and the right
   container.
5. Register a second adapter `tone-casual`. Hot-load. Verify both
   loaded.
6. Register a third (`tone-snarky`), then a fourth (`tone-clinical`),
   then a fifth (`tone-pirate`) - fifth load triggers LRU eviction
   of `tone-formal`.
7. POST with `model='tone-formal'` again - verify it gets re-loaded
   (and a different LRU adapter evicted).
8. `serve adapter rm tone-pirate --force` while it's loaded - verify
   hot-unload happens before registry deletion.

This is a long test; budget for it. It's the integration confidence
that all pieces compose.

- [ ] **Step 1**: Build the test scaffolding (mocked upstream HTTP
  with httpx ASGI transport pointing at a fake "engine" router that
  records calls).
- [ ] **Step 2**: Drive the 8-step flow.
- [ ] **Step 3**: Assert call sequences on the mocked engine.

---

## Task 12: Verification + commit

- [ ] **Step 1**: `uv run --quiet pytest tests/ -q --ignore=tests/integration`
  - all unit tests green.
- [ ] **Step 2**: `uv run --quiet pytest tests/integration -q`  -
  integration test green.
- [ ] **Step 3**: `uv run --quiet ruff check src/ tests/` - clean.
- [ ] **Step 4**: Live verification on the operator's box:
  - Pick a small base + a real public LoRA from HF (suggested:
    a Qwen2.5-0.5B + a 1-LoRA finetune from a `peft`-style repo).
  - `serve adapter pull <repo> --base qwen-0_5b`
  - `serve run qwen-0_5b --max-loras 4`
  - curl `/v1/chat/completions` with `model=<adapter-name>`
  - Capture sub-second swap latency between two adapter requests.
- [ ] **Step 5**: Commit per the conventional-commits style used in
  recent commits. Single feature commit; sub-tasks documented in the
  body.

---

## Decisions deferred to implementation discovery

These are flagged in the design as "make as you go":
- **Exact SGLang `--enable-lora-*` flag name** - verify against v0.5.x
  in container before locking in.
- **vLLM `--max-lora-rank` default** - engines have a per-rank cap;
  pick a sensible default (likely 64) and surface as `--extra` if
  needed.
- **Adapter download timeout** - same as base model download (no
  timeout); HF handles resume on its own.
- **What happens to `deployment_adapters` rows on deployment failure
  mid-load** - clean them up in the failure path (parallel to how the
  existing `dep_store.update_status('failed')` path works).

## Out of scope (Workstreams B/C/D)

- Snapshot-based fast load (B)
- Predictive pre-warm of adapters (C)
- Adapter sync between peers (D)
- UI for adapters (separate v2 UI deepening work)
