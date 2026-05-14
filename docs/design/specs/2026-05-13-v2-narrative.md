# serve-engine v2 — Narrative

**Status:** Draft, brainstorming output
**Date:** 2026-05-13
**Working name:** v2.0 (final tag TBD; could ship as 0.2.0 / 1.0.0 / etc.)

> Superseded direction note: this document is retained as historical
> brainstorming. The current product direction is the service router and
> lifecycle control plane described in
> `docs/design/specs/2026-05-14-service-router-control-plane.md`.

## 1. Where v1 ended

v1 shipped a single-node, multi-model, multi-user inference orchestrator
over vLLM, SGLang, and TensorRT-LLM. It is the solution to the problem
*"I have one beefy GPU box and I want it to serve many models and many
users without writing the orchestration scaffolding myself."*

Verified end-to-end on H100 and RTX PRO 6000 Blackwell:
- 3 engines pluggable behind one OpenAI-compatible API
- Pinned + LRU-evictable model lifecycle, KV-aware GPU placement
- API keys with 8-window sliding rate limits
- Web UI with dashboard, models, playground, keys, logs
- Live observability — Prometheus metrics aggregated across engines
  (TRT-LLM JSON translated transparently), SSE event stream, GPU stats
- Auto target_concurrency from model architecture (no more 30B-default
  footgun for 0.6B models — auto-picks 73 for Qwen3-0.6B at ctx=2048)
- 180 unit tests, ruff clean

The shape of v1 is solid. v2 is not a rewrite. It is a deliberate
expansion along axes the v1 spec explicitly deferred or didn't anticipate.

## 2. The v2 vision in one sentence

**Many boxes, one logical orchestrator, with a model lifecycle that
predicts what's needed and gets it warm before you ask.**

Three things change about what serve-engine *is*:

1. It stops being a single-box tool. Multiple boxes federate into one
   logical fleet sharing a registry. (NOT multi-node inference — engines
   still run as one container per model on one box. Federation is about
   the orchestrator layer, not the inference layer.)
2. The lifecycle becomes proactive, not reactive. The daemon learns
   usage patterns and pre-warms models before requests land.
3. Adapters (LoRA / DoRA) become first-class. "A model" is no longer
   monolithic — the cheap, fast, common operation becomes adapter
   hot-swap, not full model load.

## 3. Sub-projects

v2 decomposes into four sub-projects with a strict dependency order.
Each ships independently as a meaningful release; the whole arc is
v2.0.

### Sub-project A: Adapter-first lifecycle  *(branch `feat/v2-loading`, ships first)*

**Adapters as first-class entities.** A LoRA adapter is registered
separately from its base model and identified by name. Clients call
`model='qwen3-7b:my-adapter-v3'` (or `model='my-adapter-v3'` resolving
to its base). The proxy dispatches to a deployment that has both the
base and the adapter loaded, hot-swapping adapters per request via
vLLM's `--enable-lora` / SGLang's `--lora-paths`.

**Critical decisions (locked in):**
- Addressing: separate first-class entity. Adapters appear in
  `/v1/models` alongside base models with a `base:` field pointing to
  their parent.
- Routing: client says `model=adapter_name`. Proxy looks up the adapter,
  finds (or causes to load) a deployment of its base with this adapter
  loaded, dispatches there.
- Eviction: bases evict at full-deployment granularity (LRU, same as
  v1). Adapters within a deployment evict per-adapter (LRU within the
  base's slot budget). An adapter "swap" is sub-second; it does not
  trigger base eviction.
- Federation (deferred): adapter blobs (typically 10-200 MB) will
  replicate via on-demand pull from the peer that has them once
  Sub-project D lands. For Sub-project A we design the registry shape
  to be federation-ready (stable adapter IDs, source tracking) but
  don't implement the sync.

**Success criteria:**
- Register a base + N adapters for it. Mixed inbound traffic dispatches
  to the right adapter with sub-second per-request adapter swap latency.
- vLLM and SGLang both work; TRT-LLM gets a clear "adapter unsupported
  on this engine" error (TRT-LLM's adapter story is thinner; deferred).
- `serve adapter add/ls/rm` CLI ergonomics match the existing model CLI.

**Why first:** Highest-leverage, lowest-risk feature. Engines support
this natively; the work is mostly orchestration. Smallest "new shape"
against the existing v1 lifecycle. Ships meaningful value on day one
to single-box users.

### Sub-project B: Snapshot-based fast loads  *(branch `feat/v2-loading`)*

**Engine-state snapshot/restore.** Once a model is loaded, snapshot the
engine's CUDA-side state to disk (per-engine format). On subsequent
loads of the same (model, engine version, GPU arch, quantization,
deployment shape), restore from the snapshot in seconds instead of
cold-loading in 30-120 seconds.

**Critical decisions (locked in):**
- Storage: local disk per box. Each box stores snapshots it created at
  `~/.serve/snapshots/<key>/`. Federation pull-on-demand layered on
  later by Sub-project D.
- Snapshot key: SHA-256 of (hf_repo, revision, engine, engine_version,
  gpu_arch, quantization, max_model_len, dtype, target_concurrency).
  Two deployments with the same key share a snapshot.
- Engine support: ship for whichever engines actually support it well.
  Investigation work is part of the sub-project. Initial expectation:
  vLLM via torch.compile cache + tensor warmup; TRT-LLM has a
  serialized-engine path that's the AOT-compile build (different from
  the PyTorch backend we use); SGLang's snapshot is experimental as of
  v0.5.x.

**Success criteria:**
- For at least one engine, second-load of a model is ≥5× faster than
  first-load.
- Snapshots have a clear GC story (configurable disk quota; LRU
  eviction within quota).

**Why second:** Biggest engineering risk (engine internals, varying
support quality). Benefits from the adapter lifecycle being stable
first (so we don't re-snapshot when adapters change).

### Sub-project C: Predictive layer  *(branch `feat/v2-loading`)*

**The daemon learns your usage.** Every request writes a usage row
(model, adapter, key, timestamp, tokens). The predictor mines this for
patterns — time-of-day, sequencing ("requests for model A are usually
followed within 30s by requests for model B"), key affinity ("api-key-X
mostly hits model Y"). The lifecycle uses these predictions to
proactively pre-warm models before requests arrive, and to bias eviction
toward models that historical patterns say won't be needed soon.

**Critical decisions (defaults — to be revisited closer to ship):**
- Learner: rule-based to start (time-of-day buckets, sequencing rules,
  key→model affinity). Easy to debug and explain. ML-based classifier
  can layer on later if rule-based isn't winning enough.
- Storage: usage rows in the existing sqlite registry. Rows GC after
  a configurable retention window (default: 30 days).
- Acting on predictions: "warm in background" hooks into the lifecycle
  manager. Predictions are advisory — the user's `--pin` always wins;
  active eviction respects in-flight requests.
- Federated demand (Sub-project D): once federation lands, predictions
  span boxes. For C we design the usage schema federation-ready.

**Success criteria:**
- Recorded a representative workload trace for ≥1 week. Replaying it
  with predictions on shows ≥30% fewer cold-load events than v1's pure
  LRU.
- Predictions are explainable: `serve predict` shows what the daemon
  thinks is coming next and why.
- Bad predictions degrade gracefully: cold loads still happen on miss;
  no user-visible failures from a wrong prediction.

**Why third:** Highest sophistication. Depends on usage data the other
sub-projects accumulate. Most user-perceptible — predictive pre-warming
is the "magic" moment, but it's only magic if the underlying mechanics
(adapter swaps, snapshot restore) are reliable.

### Sub-project D: Federation primitive  *(branch `feat/v2-federation`, off `feat/v2-loading`)*

**The substrate for many boxes.** Box discovery, peer-to-peer auth,
gossiped registry of (models, adapters, deployments, usage history,
snapshot index), transparent cross-box request forwarding through the
OpenAI proxy.

**Critical decisions (locked in):**
- Trust: PSK in `~/.serve/federation.yaml`. Each peer carries the same
  shared secret. Rotation is manual coordination. No PKI.
- Consistency: eventually consistent HTTP-push gossip. Each box owns
  its local rows, pushes updates to peers periodically and on change.
  Conflicts resolved by `(timestamp DESC, peer_id ASC)`. No Raft.
- Discovery: config-only for v2.0. Peers listed by URL in
  `~/.serve/federation.yaml`. mDNS / auto-discovery deferred to v2.x.
- Forwarding: proxy mode (request hits boxA → boxA proxies to boxB if
  boxB has the model warm). Client sees one URL.

**Success criteria:**
- Two boxes can be configured to peer with each other in under 5 minutes
  using only the CLI and a yaml file.
- A model registered on boxA appears in `serve ls` on boxB within 10s.
- A request hitting boxA for a model only loaded on boxB returns
  successfully and transparently (client sees no difference from a
  same-box request beyond latency).
- Network partition between boxes: each side keeps serving locally.
  When the partition heals, registries reconcile via
  last-write-wins. No daemon crash.

**Why last:** Federation only pays off once the operator has ≥2 boxes
to federate. Sub-projects A/B/C deliver value to the single-box user
on day one. Federation also benefits from being designed AFTER the
loading sub-projects ship, because it gets to federate the actual data
shapes that exist instead of designing in the abstract. (Cost: the A/B/C
schemas need additive `peer_id`/`source_peer_id` columns when D lands.
Bounded retrofit; explicitly accepted in §4.)

## 4. Sequencing

**Reordered 2026-05-13 mid-session.** Federation was originally going
to land first as a "foundational constraint." Pragmatic reread: loading
improvements (adapters, snapshots, predictive) deliver value on a
single box from day one; federation only pays off once you have ≥2
boxes. Ship single-box value first; federate when the foundations are
real.

```
v1 (today)            ← branch: main
   │
   ▼
Branch: feat/v2-loading
   │
   ├──▶ Sub-project A: Adapter lifecycle    ← v2.0-alpha
   │       │
   │       ▼
   │     Sub-project B: Snapshot system     ← v2.0-beta
   │       │
   │       ▼
   │     Sub-project C: Predictive layer    ← v2.0-rc
   │
   ▼
Branch: feat/v2-federation  (off feat/v2-loading)
   │
   ▼
Sub-project D: Federation primitive         ← v2.0
```

The cost of this reorder is some retrofit pain when federation lands —
data structures designed for one box need `peer_id` / `source_peer_id`
columns added, sync hooks added, etc. That cost is real but bounded
(<1 week of work) and explicitly accepted in exchange for shipping
single-box value first.

Total scope: realistically 3-6 months of focused work. Each sub-project
is its own design + plan + implementation cycle, ships as a discrete
release, and is independently useful even if the next sub-project never
lands.

## 5. Non-goals (still)

The v1 spec's non-goals largely carry forward unchanged:

- **Multi-node inference.** Federation is the orchestrator federating;
  individual model deployments still run as one container on one box.
  Multi-node tensor-parallel via NCCL across boxes is out of scope.
- **Fine-tuning or training.** Adapters are loaded for inference, not
  trained. Adapter creation is a separate problem.
- **Built-in billing / invoicing.** Usage data feeds the predictor;
  it's not a billing system.
- **RBAC beyond admin + API keys.** Multi-tenant teams / orgs / roles
  are not in v2.
- **Custom audit logging beyond structured stdout.** Compliance-grade
  audit trails would be a separate product.
- **Replacing the engines.** vLLM, SGLang, TRT-LLM keep doing the
  inference. We orchestrate, federate, and predict; they generate.

New v2 non-goals:

- **Cross-engine adapter compatibility.** A LoRA trained for vLLM
  loading is loaded by vLLM; we don't translate adapter formats
  between engines.
- **Snapshot portability across GPU architectures.** A snapshot built
  for sm_90 (H100) is not restored on sm_120 (Blackwell). Snapshot key
  includes gpu_arch precisely to avoid this mistake.
- **Geographic distribution.** Federation works on a LAN or low-latency
  WAN. We are not solving high-latency cross-region replication.

## 6. Backward compatibility

v2 should not break v1 *clients*. The OpenAI-compatible API surface
stays. The `serve` CLI grows new subcommands (`serve peers …`) but
existing ones keep working. The HTTP admin API gains new routes under
`/federation/*` and `/v1/adapters/*`; v1 routes are unchanged.

v2 *daemons* with no federation peers configured behave exactly like
v1 daemons. Federation is opt-in via the existence of
`~/.serve/federation.yaml` with at least one peer entry.

The sqlite schema gains additive columns (`source_peer_id`, `updated_at`
on the registry tables; new `peers`, `adapters`, `usage` tables).
Forward migrations are idempotent. v1 → v2 upgrade does not require
data migration; the v2 daemon reads the v1 schema, applies the
additive migrations on first start, and continues.

## 7. What's explicitly NOT decided yet

These need user input before / during the relevant sub-project, and
are flagged so they don't get silently chosen by the implementation:

- **Federation discovery beyond config.** mDNS? Service catalog? Cloud
  metadata? — defer to v2.x.
- **Adapter format support matrix.** LoRA always; QLoRA, DoRA,
  IA³ — confirm per engine.
- **Snapshot eviction policy.** Snapshot directory will grow without
  bound; need a GC strategy. Disk quota? LRU? Tied to model eviction?
- **Predictive learner sophistication.** Rule-based is the v2.0 default.
  Whether we ever ship an ML classifier (and if so, what model) is open.
- **Federation observability.** Prometheus metrics on peer health,
  registry sync lag, cross-box request rate, adapter pull latency. Not
  designed yet; should be part of sub-project 1.
- **CLI for federation.** Provisional: `serve peers add/ls/rm/ping`,
  `serve --peer <name> <command>` for explicit cross-box ops. Will
  refine in sub-project 1's design.

## 8. Open questions for the v1 → v2 transition

- **Versioning scheme.** v1 was internal; v2 is the first public-ish
  release. Pick semver (0.x → 1.x → 2.x) vs. a fresh start (1.0).
- **Public release timing.** Original brainstorm pairing was UI
  deepening + public release. v2's federation primitive is heavier
  than that. Decide whether to ship a v1.5 polish-and-release first,
  or roll all of v2 into the first public moment.
- **Documentation.** v2 changes the operator mental model
  significantly (boxes → fleet, models → models+adapters). Docs need
  to grow with it.
