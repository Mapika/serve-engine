# UI Thesis: Sequencing Plan

> **For agentic workers:** Each numbered view below has its own detailed implementation plan (or will when it comes up). This document is the *order* — why we tackle them in this sequence, what depends on what.

**Goal:** Make the bundled web UI the differentiating surface of serve-engine.

**Architecture:** Six independent views, each one self-contained. Three of them need no backend changes (gap-fills against existing `/admin/*` endpoints). The other three add small backend endpoints in service of specific UI views.

**Tech Stack:** React 18, Vite, Tailwind, @tanstack/react-query, FastAPI (existing daemon). No JS test runner installed; manual browser verification in dev.

---

## Sequence

### 1. Services & Routes view  *(no backend changes — pure gap-fill)*

The biggest *visible* gap. `/admin/service-profiles` and `/admin/routes` exist; the README documents them as the recommended way to use the engine; the UI exposes neither.

**Why first:** Unblocks views 2 and 4. The side-by-side playground (view 2) needs to be able to pick *routes*, not just models. The route dry-run (view 4) is an enhancement of this view.

### 2. Side-by-side playground  *(no backend changes — blocked by view 1)*

Refactor `Playground.tsx` so a prompt streams against two routes simultaneously, with a comparison row (TTFT, throughput, total ms). This is the "demo to your team" view.

**Why after view 1:** Comparing routes only makes sense if routes are visible elsewhere in the UI.

### 3. Live deployment graph  *(no backend changes)*

The current dashboard shows a table. Replace the top half with a visual GPU-card-with-stacked-deployments view that makes the VRAM sharing story (the project's actual differentiator per README §GPU Sharing) visible at runtime.

**Why third:** It's an independent visual reframe. Slots in after views 1+2 are stable.

### 4. Route dry-run  *(small backend addition — blocked by view 1)*

Add `GET /admin/routes/match?model=X`, return matched route + profile + match reason. Add a "test request" panel to the Routes view.

**Why fourth:** Builds on view 1; the smallest backend change of the views that need any.

### 5. Per-key usage charts  *(medium backend addition)*

Add `GET /admin/keys/{id}/usage` that buckets the usage table by hour. Render per-key sparklines + a detail drawer with full charts in the Keys view.

**Why fifth:** Requires understanding the existing usage table schema and time-bucketing. Bigger backend scope than view 4.

### 6. Live request inspector  *(largest scope, instruments the router)*

Per-request tracing through the router. New SSE stream `/admin/requests/stream`. Live feed + waterfall view.

**Why last:** Touches the hot path. Worth doing carefully *after* the other views have proven the visual pattern.

---

## Out of scope for this thesis

- First-run wizard polish (mentioned in earlier brainstorming) — defer until the six views above ship; the wizard is then a thin orchestration over them.
- Curated model catalog — separate product bet, not a UI bet.
- Multi-tenant economics (budgets, billing) — separate product bet.
