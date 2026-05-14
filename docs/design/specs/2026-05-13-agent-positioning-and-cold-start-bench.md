# Agent positioning + cold-start benchmark - launch design

**Date:** 2026-05-13
**Status:** Approved (Mark, 2026-05-13). Ready for implementation plan.
**Author:** Mark Marosi
**Type:** Launch / positioning + benchmark deliverable

## Goal

Give serve-engine a **distinctive pull** at open-source launch by combining two streams of work that ride on what's already built:

1. **B - Agent positioning.** Reframe serve-engine in the README and examples as *the GPU box for a small AI team running agent workloads* - not as a generic inference wrapper.
2. **A - Cold-start benchmark.** Turn the snapshot subsystem (torch.compile cache persistence) into the launch's headline measurable: a defensible "cold vs. warm" table that goes in the README, the launch blog post, and the HN front-page comment.

Together: a clear *who-this-is-for* statement (B) backed by a concrete *why-it's-better* number (A).

## Non-goals

- **No new product abstractions.** No "agent" or "model group" feature in the codebase. The agent narrative is carried by `examples/` and `README.md`, not by new primitives. (Building features to support marketing copy is the wrong order.)
- **No predictor demo / no usage-pattern story.** Option C from brainstorming is parked for a v2 launch.
- **No CI workflow, no CONTRIBUTING.md, no contributor outreach.** The brainstorming clarified that *contributor community* is explicitly not a launch goal. Operator UX is the priority.
- **No TRT-LLM smoke script in scope here.** It's a separate hygiene gap (flagged in earlier review); does not block this launch.
- **No multi-GPU / tensor-parallel benchmarks.** Single-GPU only. Multi-GPU is a future launch story.

## Deliverables

| Path | Stream | What it is |
|---|---|---|
| `README.md` (rewrite) | B | New first paragraph + reordered feature list. Opens with the agent narrative; existing capabilities listed underneath as proof. |
| `examples/README.md` (new) | B | Index of the recipes - which one demonstrates what, when to read each. |
| `examples/01-router-reasoner/` (new) | B | Small-model-routes-to-big-model demo. |
| `examples/02-rag-embed-chat/` (new) | B | Embeddings + chat on one daemon. |
| `examples/03-lora-per-task/` (new) | B | LoRA hot-load demo - the differentiator. |
| `scripts/bench_snapshots.py` (new) | A | Cold-vs-warm benchmark harness. |
| `docs/bench/snapshot-cold-vs-warm.json` (new) | A | Checked-in benchmark output. |
| `README.md` - Benchmarks section (new) | A | Cold-vs-warm table referencing the JSON. |
| `LAUNCH.md` (new) | B | Draft of the launch blog post / HN top comment, kept in repo as an editable doc. |

## Critical-path order

Numbers first, narrative second:

1. **Build `bench_snapshots.py`** -> run it -> record real numbers.
2. **Sanity-check the numbers.** If the cold-vs-warm gap isn't dramatic (>=5x on First TTFT specifically - see Success criteria), the launch story has to change before any README rewrite. Stop and reassess before continuing.
3. **Build the three examples.** Each one ends with checked-in `sample-output.txt` produced from a real run.
4. **Rewrite the README** around verified numbers + working examples.
5. **Draft `LAUNCH.md`.**

Rationale: a great README around bad numbers is worse than no README. The benchmark gates the rest.

## Section 1 - Examples (`examples/`)

### Constraints (all three recipes)

- All models **public and ungated.** No HuggingFace token required to run.
- Combined VRAM <= 24 GB per recipe so they run on a 4090/5090/A6000, not only H100.
- Each `client.py` uses **only** the `openai` Python SDK (plus `numpy`/`faiss-cpu` where unavoidable for recipe 02). **No imports from `serve_engine`.** Proves the OpenAI-compatibility claim.
- No tests for examples. They are proofs by demonstration, not regression-protected code.
- Per-recipe layout (consistent so readers learn the pattern):
  ```
  NN-<name>/
  |-- README.md          # 1 page: problem, what this recipe shows, expected output
  |-- setup.sh           # serve pull <model> ...; serve run <model> ...
  |-- client.py          # OpenAI SDK demo
  +-- sample-output.txt  # checked-in real output so readers can verify
  ```

### 01 - `router-reasoner/`

- **Models:** `Qwen/Qwen2.5-0.5B-Instruct` (router) + `meta-llama/Llama-3.2-1B-Instruct` (reasoner). (Backup if Llama gates: substitute `Qwen/Qwen2.5-1.5B-Instruct`.)
- **What `client.py` does:** sends a fixed list of 20 mixed prompts (10 trivial, 10 hard). Router model classifies each as `simple|complex`; classification result routes the request to the appropriate reasoner. Prints final answers + a summary line: `routed 10 to small (avg X tok), 10 to large (avg Y tok); est cost saved vs. always-large: Z%`.
- **What it proves:** multi-model serving on one daemon, one OpenAI endpoint, real cost win from heterogeneous routing.

### 02 - `rag-embed-chat/`

- **Models:** `BAAI/bge-small-en-v1.5` (embeddings) + `Qwen/Qwen2.5-1.5B-Instruct` (chat).
- **Corpus:** 10 short paragraphs extracted from serve-engine's own `README.md`, checked into `examples/02-rag-embed-chat/docs/`. The recipe is meta - "ask serve-engine what serve-engine is."
- **What `setup.sh` does:** pulls both models, runs both pinned.
- **What `client.py` does:** builds a FAISS index from the 10 docs (one-time at startup), then asks "what is serve-engine?" - embeds the query via `/v1/embeddings`, retrieves top-3, sends those + the query to `/v1/chat/completions`, prints the answer.
- **What it proves:** two different model families coexist; both OpenAI endpoint families work as advertised.

### 03 - `lora-per-task/`

- **Models:** one public LoRA-friendly base (decision: `Qwen/Qwen2.5-1.5B-Instruct` if a small base also works for the adapters we pick; otherwise `meta-llama/Llama-3.1-8B-Instruct`) + 2-3 public LoRAs whose target_modules match the base. **Implementation-time task: identify exact LoRAs before writing `setup.sh`.** Fallback if no good public LoRAs match: train two tiny LoRAs ourselves with `peft` and host them in a sibling HF repo.
- **What `client.py` does:** fires 3 back-to-back chat completions, each with `model=<adapter-name>`. Prints TTFT for each. First request to each adapter is the cold-load cost; subsequent same-adapter requests are sub-second.
- **What it proves:** LoRA hot-load works through a vanilla OpenAI client - the feature competitors don't have.

### Cuts (explicit YAGNI)

- No vision/multimodal recipe - adds image deps, audience fragments.
- No tool-calling recipe - engine-version brittleness.
- No multi-tenant/quotas recipe - feature exists, mentioned in README, not a wedge example.
- No CI for the examples - they're meant to be run, not guarded.

## Section 2 - Snapshot benchmark (`scripts/bench_snapshots.py`)

### Methodology

- **Pre-warmth before timing starts:**
  - Docker image already pulled.
  - HF weights already pulled (`serve pull <model>` ahead of time).
  - Daemon running, no current deployment.
  - For the cold condition: snapshot directory empty (deleted by the harness).
  - For the warm condition: snapshot present (left over from the cold run).
- **5 runs per condition.** Report median + min/max in JSON; lead with median in the README.
- **Two sub-metrics per run, reported separately:**
  - *Engine-ready time*: `POST /admin/deployments` -> status=`ready` (engine's own healthcheck).
  - *First TTFT*: `POST /v1/chat/completions` -> first token byte received (measured client-side over loopback).
  - *Total wallclock*: sum. This is the user-visible "cold start" number.
- **Snapshot save is async** (post-deployment-ready); harness waits for the snapshot-save task to settle before declaring cold-run done. Without this wait, the warm run has nothing to consume.
- **Honest framing in the README:** "First boot of any model is always cold - the snapshot is built during that first run. Every subsequent boot is warm." No pretense that the very first request is fast.

### Implementation choices

- **Transport: direct admin HTTP via `httpx`**, not subprocess `serve` CLI. We're measuring the daemon, not CLI overhead, and HTTP is less brittle than parsing CLI output.
- **Single benchmark execution generates both conditions.** Cold run produces the snapshot the warm run consumes.
- **Per-run cleanup:** stop deployment between runs; for cold runs, also `rm -rf` the snapshot dir for that key.
- **Failure modes:**
  - Daemon crash -> harness bails non-zero; daemon's own reconciliation handles cleanup.
  - GPU OOM -> caught, recorded in JSON as failure for that run, not silently dropped.
  - Snapshot save never settles (>2 min) -> that run recorded as failure; benchmark continues.

### Scope

- **vLLM only for v1.** SGLang has the same snapshot mechanism but doubles runtime; add post-launch if reception is good. README explicitly states "same applies to SGLang."
- **One model size for the headline:** `Qwen/Qwen2.5-1.5B-Instruct` (small enough for 4090, large enough that torch.compile time matters). `--model` flag exposes rerunning on any model.

### CLI

```
python scripts/bench_snapshots.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --runs 5 \
    --output docs/bench/snapshot-cold-vs-warm.json
```

### Output

- JSON file with: per-run timings, median/min/max per condition, hardware fingerprint (`nvidia-smi --query-gpu=name,driver_version,memory.total`), serve-engine git SHA.
- Stdout: a markdown table that's copy-pasteable into the README:

  ```
  | Phase                | Cold (no snapshot) | Warm (from snapshot) | Speedup |
  |----------------------|--------------------:|---------------------:|--------:|
  | Engine ready         |               X.Xs |                 Y.Ys |    Z.Zx |
  | First TTFT           |               X.Xs |                 Y.Ys |    Z.Zx |
  | Total wallclock      |               X.Xs |                 Y.Ys |    Z.Zx |
  ```

## Section 3 - README rewrite (B)

### What changes

- **First paragraph** opens with the agent narrative: *"serve-engine is the GPU box for a small AI team. Your agents call 5 models from one OpenAI endpoint - a tiny router, a medium retriever, a big reasoner, plus the LoRAs your team trained - on one machine. serve-engine keeps the hot ones warm and swaps the rest."* (No hard latency numbers in the framing copy - the cold-start table below is the proof; the framing doesn't pre-commit to specific seconds.)
- **Existing "What it does" section reordered** so the agent-relevant bullets come first: multi-model on one daemon, LoRA hot-load, OpenAI-compatible, per-user keys. Engine-pluggability and the Prometheus/SSE stuff move further down.
- **New "Cold start" subsection** with the benchmark table.
- **New "Examples" section** linking to `examples/README.md`.
- **Status line stays factual.** Current `156 unit + integration tests` already corrected to `363`; no further claim inflation.

### What does NOT change

- Architecture diagram stays. CLI table stays. On-disk-layout section stays. Quickstart stays (the agent narrative is the framing; the quickstart still works as-is).

## Section 4 - `LAUNCH.md`

A repo-internal draft of:

1. A 200-word "what this is and why it exists" blurb (HN top-of-comment material).
2. A 600-word blog-post draft expanding on the same. Mostly: the agent-team problem, the cold-start numbers, the LoRA hot-load demo. Honest about what's *not* there yet (no multi-node, no autotune, no contributor community plan).
3. A checklist of launch-day actions (paste-where, links to verify, etc.).

Kept in-repo so it's reviewable and improves over time, not lost in Notion.

## Success criteria (definition of done)

- `bench_snapshots.py` runs end-to-end on the developer's available single-GPU hardware (RTX PRO 6000 Blackwell or H100) and produces both a JSON file and the stdout table. **The "First TTFT" row** must show >=5x speedup (cold vs. warm). If <5x, design pauses for reassessment; this is not a rubber-stamp deliverable.
- All three recipes runnable as `cd examples/NN-... && ./setup.sh && python client.py`, producing output that matches the checked-in `sample-output.txt` (within reasonable variance - first-tier-token strings differ run-to-run, but the program completes successfully).
- README rewrite landed; first paragraph is agent-narrative; cold-start table embedded; examples linked.
- `LAUNCH.md` draft committed.
- All 363 existing tests still pass; ruff still clean.
- No new feature flags, new abstractions, or schema migrations introduced. (If implementation reveals one is needed, design returns to brainstorming for that piece.)

## Open implementation questions

1. **LoRA selection for recipe 03.** Need to identify 2-3 public, non-gated LoRAs whose target_modules match the chosen base. If none exist, train tiny ones with `peft` and host in a sibling HF repo under the author's namespace. Resolved during implementation, not blocking design.
2. **bge-small-en-v1.5 on vLLM/SGLang.** Verify the embedding model serves correctly through serve-engine's existing OpenAI proxy at `/v1/embeddings`. If proxying is incomplete for embeddings, that's a real bug to fix or a recipe substitution. Should be quickly verifiable during implementation.
3. **Llama 3.2 gating.** If the gated Llama model in recipe 01 is a friction point, the Qwen fallback should be the default to keep the recipe friction-free.

These are all implementation-time clarifications, not design changes.

## Risks

- **Snapshot numbers underwhelm.** Mitigation: run the benchmark first (critical-path step 1). If the gap is <5x we reassess before any other work.
- **Recipe 03 LoRAs are hard to source.** Mitigation: training tiny LoRAs ourselves with `peft` is ~1 day of work and ensures we control the demo.
- **Examples drift from the daemon.** Mitigation: each `sample-output.txt` is checked in from a real run. If the recipes break later, the diff in expected output is the canary. Not CI, but visible.
- **README rewrite invalidates existing positioning.** Mitigation: the existing capabilities aren't removed - only reordered. The "single-node multi-user inference orchestrator" line still fits below the agent paragraph as a one-liner subtitle.
