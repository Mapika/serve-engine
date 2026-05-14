# Serving Engine - Plan 08: Hardening

**Goal:** address the 9 issues + 4 nits surfaced when reviewing the system after Plan 07. Minimize new code; prefer fixes over rewrites.

---

## Issue -> fix table

| # | Issue | Files touched | LoC |
|---|-------|---------------|-----|
| 1 | Daemon crash -> orphaned engine containers | `daemon/app.py`, `lifecycle/manager.py` | ~30 |
| 2 | Proxy drops upstream status code & headers | `daemon/openai_proxy.py` | ~30 |
| 3 | `_extract_usage` buffers entire response | `daemon/openai_proxy.py` | ~15 |
| 4 | `serve pull` doesn't actually pull weights | `cli/pull_cmd.py`, `daemon/admin.py` | ~30 |
| 5 | Web UI Logs view broken under auth | `ui/src/views/Logs.tsx`, `daemon/admin.py` | ~30 |
| 6 | No `serve update-engines` | `cli/update_engines_cmd.py` (new), `cli/__init__.py` | ~40 |
| 7 | `backends.yaml` is dead - image tags hardcoded | `backends/base.py`, `backends/vllm.py`, `backends/sglang.py`, `daemon/__main__.py` | ~40 |
| 8 | Reaper time-eviction never tested live | `tests/unit/test_reaper.py` (a real timing test) | ~30 |
| 9 | Hardcoded vLLM-headroom constants | `backends/backends.yaml`, `lifecycle/manager.py` | ~30 |
| Nit | `pynvml` deprecation | `pyproject.toml` | ~2 |
| Nit | `ipc.delete` error detail | `cli/ipc.py` | ~6 |
| Nit | Real-HTTP integration test for proxy | `tests/integration/test_openai_proxy_http.py` (new) | ~80 |

Total target: ~360 LoC across ~12 commits.

## Order of work

Issues that change shared surfaces first, dependent fixes after.

**Phase 1 - Backend manifest plumbing (7 + 9):**
- Load `backends.yaml` once at daemon startup.
- `Backend` Protocol gets a `headroom` config object.
- vLLM/SGLang classes read `image_default` + `internal_port` from manifest.
- `manager.load` reads headroom constants from `backend.headroom`.

**Phase 2 - Daemon reliability (1):**
- FastAPI lifespan replaces `on_event` hooks (already deprecated).
- On shutdown: `manager.stop_all()` synchronous.
- On startup: `manager.reconcile()` - walks ready rows, marks orphans `failed`, optionally re-adopts via `docker inspect`.

**Phase 3 - Proxy correctness (2 + 3):**
- Enter `client.stream(...)` context outside the streamer to read status+headers.
- Pass them into `StreamingResponse(status_code=..., headers=...)`.
- `_extract_usage` rewritten to parse only the last `data:` chunk for `usage`. Keep last-chunk buffer ~4KB.

**Phase 4 - UX fixes (4):**
- `serve pull` now downloads weights via a new `POST /admin/models/{name}/download` admin endpoint that calls `download_model` in a background task; CLI streams a progress dot every ~2s until done.

**Phase 5 - UI auth fix (5):**
- Add `GET /admin/events?token=<sk-...>` query-param auth so `EventSource` works. Falls back to header if present.

**Phase 6 - Engine updates (6):**
- `serve update-engines` resolves latest stable tags from Docker Hub for each backend in `backends.yaml`, writes a new YAML to `~/.serve/backends.override.yaml`. Daemon prefers override on startup.

**Phase 7 - Test + nit cleanup (8 + nits):**
- Real-HTTP reaper test with `time.sleep` over a short window.
- pyproject: replace `pynvml>=11.5` with `nvidia-ml-py>=12.0` (compatible API).
- `cli/ipc.delete` mirrors the JSON detail extraction of `get`/`post`.
- New integration test using a real HTTP server (not ASGITransport) to exercise the proxy.

---

## Verification

1. `pytest -v` -> all current tests pass + new tests added.
2. `ruff check src/ tests/` clean.
3. Live: kill daemon mid-load -> restart -> confirm orphan recovery.
4. Live: hit a non-existent endpoint on the engine -> confirm proxy returns 404, not 200.
5. Live: `serve pull` actually downloads.
6. Live: `serve update-engines --dry-run` lists newer tags if available.

That's the whole plan. No new abstractions, no expanded scope.
