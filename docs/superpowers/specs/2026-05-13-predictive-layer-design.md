# Sub-project C — Predictive Pre-warming: Design

**Status:** Draft, ready for review (written ahead of implementation
overnight 2026-05-12 → 2026-05-13)
**Date:** 2026-05-13
**Branch:** `feat/v2-loading` (lands after Sub-projects A and B)
**Prerequisites:** Sub-projects A (adapters) and B (snapshots) complete
**Companion docs:** `2026-05-13-v2-narrative.md`,
`2026-05-13-adapter-lifecycle-design.md`,
`2026-05-13-snapshot-system-design.md`

## 1. Goal

The daemon learns your workload patterns and pre-warms models /
adapters before requests arrive. Goal: ≥30% reduction in cold-load
events vs v1's pure-LRU eviction on a representative production trace.

The user-perceptible difference is "the model I asked for was already
warm" instead of "I waited 30s for it to load." The mechanism is
boring rule-based prediction over a usage history table; the magic is
that it's continuously running, federated (when D lands), and tied
into the snapshot system from B so warm-up is cheap.

## 2. Non-goals

- **Not:** training an ML model for prediction. v2.0 ships rule-based
  prediction (time-of-day buckets, sequencing rules, key→model
  affinity). ML classifier may layer on later if rule-based isn't
  winning enough.
- **Not:** prediction across boxes in v2.0. Federated demand awareness
  is part of Sub-project D.
- **Not:** speculative inference (running the same request on multiple
  models for latency hedging). Out of scope; engine-internals concern.
- **Not:** changing the user's `--pin` semantics. Pinned models stay
  pinned regardless of predictions.
- **Not:** learning client identity beyond API key (no IP-based
  fingerprinting, no header inspection beyond the auth key).

## 3. Schema additions

```sql
-- Sub-project C (v2): per-request usage events for prediction.
CREATE TABLE IF NOT EXISTS usage_events (
    id            INTEGER PRIMARY KEY,
    ts            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    api_key_id    INTEGER REFERENCES api_keys(id),
    model_name    TEXT NOT NULL,        -- as the client said it (could be adapter)
    base_name     TEXT NOT NULL,        -- the resolved base model
    adapter_name  TEXT,                 -- the resolved adapter, NULL if none
    deployment_id INTEGER REFERENCES deployments(id),
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    cold_loaded   INTEGER NOT NULL DEFAULT 0,  -- 1 = this request triggered a load
    -- Federation-ready (Sub-project D will populate):
    source_peer_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts);
CREATE INDEX IF NOT EXISTS idx_usage_base_ts ON usage_events(base_name, ts);
CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_events(api_key_id, ts);
```

Storage cost: ~150 bytes per event. 1k req/min × 60 × 24 × 30 days =
~6 GB worst case at the high end. With default 30-day retention plus
opportunistic compaction (aggregating week-old events into a
`usage_aggregates` table), expected steady-state ~500 MB.

## 4. The predictor (rule-based v2.0)

Three rules feeding one queue of "warm-this-now" candidates:

### Rule 1: Time-of-day affinity
Bin the past 30 days into 168 (24×7) hour-of-week buckets. For each
bucket, count base/adapter activations per (base, adapter). A
candidate's score is its activation rate in the upcoming bucket. Top
K candidates pre-warm.

### Rule 2: Sequencing
Mine pairs (X → Y) where Y is loaded within T seconds of a request
to X. If `P(Y | X within T)` exceeds threshold (default 30%), Y becomes
a candidate when X is requested. Computed offline via background job
(every hour).

### Rule 3: Key affinity
For each API key, the top 5 most-used (base, adapter) pairs over the
past 7 days. When a key fires after a long idle, pre-warm its top.

The three rules combine via OR (any rule firing makes a candidate);
score is `max(rule_scores)`; the queue dedup-and-sorts.

## 5. Acting on predictions

A new `Predictor` long-running task in the daemon's lifespan, ticks
every 30s:

```python
async def predictor_tick():
    candidates = predictor.candidates(now=time.time())
    for c in candidates[: MAX_PREWARM_PER_TICK]:  # default 2
        if not is_already_warm(c):
            if has_room_to_warm(c):
                # Trigger a normal load via the lifecycle manager
                await manager.preload(c.base, c.adapter)
            elif evict_candidate := pick_eviction_victim(c):
                # Bias eviction: kick out the LRU model that prediction
                # says won't be needed soon, in favor of the predicted one
                await manager.evict_for_preload(evict_candidate, c)
```

Hard guardrails:
- Never preempt a request in flight.
- Never evict a pinned deployment.
- Cap preload-triggered loads at `MAX_PREWARM_PER_TICK` (default 2)
  so a runaway predictor can't churn the GPU.
- Predictions are advisory; if a real request comes in for a model
  the predictor was about to evict, the request wins.

## 6. The `cold_loaded` signal

`usage_events.cold_loaded = 1` if the request triggered a model or
adapter load. The proxy code already knows this (it's the difference
between "deployment found ready" and "had to call manager.load"); we
plumb it through to the usage logger.

This is the metric we optimize: `sum(cold_loaded) / count(*)` over a
representative window. v1 baseline; with predictor on we want this to
drop ≥30%.

## 7. CLI surface

```
serve predict
   # Show what the predictor thinks is coming next + reason.
   # Output:
   #   qwen3-0_6b              score=0.85 reason="time-of-day (next hour:
   #                              loaded 142x in past 30d)"
   #   qwen3-0_6b:tone-formal  score=0.62 reason="sequencing (P=0.71)
   #                              after qwen3-0_6b within 30s"

serve predict --history
   # Show recent prediction decisions: timestamp, action (warm/evict),
   # candidate, outcome (loaded/skipped/already-warm).

serve predict --replay <log-file>
   # Take a usage_events dump (e.g. exported from production), replay
   # against the predictor, report cold-load reduction vs LRU baseline.
   # Useful for tuning thresholds before enabling in production.
```

## 8. Configuration

```yaml
# ~/.serve/predictor.yaml (defaults shown)
enabled: true
tick_interval_s: 30
max_prewarm_per_tick: 2
retention_days: 30
rules:
  time_of_day:
    enabled: true
    weight: 1.0
  sequencing:
    enabled: true
    weight: 1.0
    window_s: 30
    min_p: 0.30
  key_affinity:
    enabled: true
    weight: 1.0
    top_k_per_key: 5
    idle_seconds: 300
```

## 9. Integration with snapshots (Sub-project B)

When the predictor decides to pre-warm, the lifecycle manager's load
path automatically takes the snapshot path if a matching snapshot
exists (already implemented in B). So predictive warm of a model
that's been seen before is fast (snapshot restore), while warm of a
fresh model is slow (cold load).

This means the predictor's value is highest when:
- Model has a snapshot AND
- Predictor is right about it being needed

If the predictor is wrong, we paid a few-second snapshot restore for
nothing. Acceptable cost.

## 10. Federation (Sub-project D will use)

Schema column: `usage_events.source_peer_id`. When D lands, peers
gossip their recent usage events, the predictor mines a federated
view, and "preload on box B because box A's traffic predicts demand"
becomes possible.

For sub-project C: `source_peer_id` is always NULL.

## 11. Testing strategy

- `test_predictor_rules.py` — each rule in isolation. Synthesize a
  usage_events trace, run the rule, assert the candidate set.
- `test_predictor_combine.py` — three rules together; OR semantics;
  score is max; dedup correctness.
- `test_predictor_tick.py` — tick respects guardrails: max per tick,
  pinned protection, in-flight protection.
- `test_predictor_replay.py` — replay-mode harness; given a sample
  trace, verify cold-load count under predictor-on vs baseline LRU.
- Live verification: record real workload for ≥1 day, run
  `serve predict --replay <log>`, expect ≥30% cold-load reduction.

## 12. Decisions I'm flagging for review

- **Hourly buckets vs higher resolution.** 168 buckets/week is
  coarse-grained but interpretable. Quarter-hourly = 672 buckets =
  needs more data to be statistically meaningful. Lean coarse for v2.0.
- **Sequencing window length.** Proposal: 30s. Wider = more pairs but
  more noise. Configurable (rules.sequencing.window_s).
- **Idle threshold for key-affinity rule.** Proposal: 5 minutes since
  a key's last request. Below this, no need to pre-warm; the key is
  still active.
- **Predictor on by default?** Proposal: YES. Defaults are
  conservative (2 preloads/tick, 30s tick, 30% sequencing threshold)
  so a fresh box won't aggressively preload until it has enough
  history. Operators can disable in `predictor.yaml`.
- **Aggregation strategy for old events.** Proposal: keep raw events
  for 30 days, then nightly aggregate (model, hour-of-week, count) into
  `usage_aggregates`, drop the raw rows. Keeps the predictor data
  bounded.
- **Whether to record `cold_loaded` retroactively.** The proxy knows
  at dispatch time whether the deployment was already warm. But what
  if a hot-load was triggered by the predictor 5s before the request
  arrived — is that "cold" or "warm"? Proposal: "warm" (the predictor
  succeeded). cold_loaded=1 only if the request itself blocked on
  load.

## 13. What's intentionally NOT in scope

- Federated demand prediction (Sub-project D)
- ML-based predictor (rule-based first; ML if rules don't suffice)
- Per-tenant prediction beyond API-key affinity
- Speculative inference / latency hedging across models
- Adversarial robustness (operator's API keys are trusted)
- Web UI for predictor (lands with v2 UI work)

## 14. Success criteria recap

- `sum(cold_loaded) / count(*)` drops ≥30% vs v1 on a representative
  workload trace.
- Predictions are explainable via `serve predict`; operator can see
  why each candidate was picked.
- Bad predictions degrade gracefully: at most 2 wasted preloads per
  30s (`max_prewarm_per_tick`); no user-visible failures from a wrong
  prediction.
- Daemon CPU overhead from the predictor tick is <1% on a single
  core. (Predictor queries are over indexed sqlite columns; should be
  microseconds even at million-row scale.)
