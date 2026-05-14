# serve-engine — Service Router Control Plane

**Status:** Accepted direction
**Date:** 2026-05-14

## Thesis

serve-engine is a high-performance router and lifecycle control plane for
local inference services.

It gives teams one OpenAI-compatible front door while it starts, stops,
health-checks, routes to, and supervises heterogeneous backends such as
vLLM, SGLang, TensorRT-LLM, llama.cpp, embedding services, rerankers, and
custom HTTP servers.

The product is not "LoRA serving" and it is not another model runner. The
product is the operational layer around inference services on owned
compute.

## Why This Is Different

- Compared to raw vLLM/SGLang/TRT-LLM, serve-engine owns lifecycle,
  health, routing, auth, quotas, observability, and operator workflow.
- Compared to LiteLLM-style gateways, serve-engine can start and stop the
  local services it routes to and reason about GPU placement and health.
- Compared to Ollama-style local runners, serve-engine is multi-user,
  policy-driven, engine-agnostic, and operations-focused.
- Compared to Kubernetes-first stacks, serve-engine targets the common
  single-node and small-fleet GPU operator path with less platform
  overhead.

## Core Concepts

**Service**

A runnable inference unit. It has a health endpoint, routeable address,
capacity, capabilities, logs, metrics, and lifecycle state. A service may
serve chat completions, completions, embeddings, reranking, image models,
or a custom API.

**Service profile**

A saved launch definition: backend, image or command, args, environment,
health path, routeable base path, GPU requirements, concurrency limits,
and optional capabilities.

**Backend**

The implementation adapter for a family of services. Current backends are
vLLM, SGLang, and TensorRT-LLM. Future backends can include llama.cpp,
TEI, Infinity, custom HTTP, or internal services.

**Driver**

The mechanism that starts and stops a service. Docker is the first driver.
Process, remote HTTP, SSH, Kubernetes, or Slurm drivers can follow behind
the same lifecycle interface.

**Deployment**

A concrete running instance of a service profile. It has runtime state:
`pending`, `loading`, `ready`, `stopping`, `stopped`, or `failed`.

**Route**

A mapping from an incoming request to one or more candidate deployments.
The first route family is OpenAI model-name routing. Later route families
can include endpoint path, tenant, API key, service capability, or custom
headers.

**Policy**

The decision logic after route matching. Examples: health-gated, warm-only,
least-loaded, queue-aware, fallback, canary, shadow, priority, or tenant
isolation.

**Capability**

An optional feature of a service: streaming, chat, completions, embeddings,
rerank, dynamic adapters, metrics, token accounting, or structured output.
LoRA support lives here as a backend capability, not as the product center.

## Current Mapping

The current codebase already contains most of the first service-control
plane pieces:

- `models` are today's route labels and weight registry.
- `deployments` are today's concrete running service instances.
- `backends` are service backend implementations.
- `LifecycleManager` is the lifecycle controller.
- `openai_proxy` is the first router data path.
- Docker is the only start/stop driver.
- Dynamic adapters are implemented as a backend capability.

The next architectural step is to make service profiles and route rules
explicit without breaking the existing model/deployment commands.

## Feature Roadmap

### Phase 1: Explicit Service Profiles

- Add a `service_profiles` table and API.
- Let existing `models` continue to work as a compatibility layer.
- Store launch config once: backend, image, args, env, health path,
  route base, GPU hints, and concurrency limits.
- Expose profiles in the UI as the primary launch surface.

Initial API surface:

- `GET /admin/service-profiles`
- `POST /admin/service-profiles`
- `GET /admin/service-profiles/{name}`
- `POST /admin/service-profiles/{name}/deploy`
- `DELETE /admin/service-profiles/{name}`

The first implementation stores model-serving launch definitions and
deploys them through the existing `DeploymentPlan` path. That gives the
router direction a concrete persistence/API layer without breaking the
current model/deployment compatibility surface.

### Phase 2: Router Policy Layer

- Introduce a route table separate from the model registry.
- Add health-gated route selection.
- Add fallback routes for cold, failed, overloaded, or missing services.
- Add least-loaded routing where multiple ready deployments can satisfy a
  route.

### Phase 3: Backpressure

- Track in-flight request count and queue depth per deployment.
- Enforce per-service concurrency caps.
- Add bounded queues with queue timeout and clear `429`/`503` semantics.
- Record queue time as a first-class latency metric.

### Phase 4: Autosuspend And Autostart

- Generalize idle timeout into service policy.
- Optionally start a stopped service on first request.
- Queue briefly while the service starts, then forward or time out.
- Keep pinning as the explicit "never autosuspend" operator control.

### Phase 5: Broader Service Families

- Add embeddings and rerank service profiles.
- Add custom HTTP services with configurable health and route paths.
- Add a process driver for non-Docker local services.
- Add remote service registration for services the daemon should route to
  but not start.

## Non-Goals

- Not a replacement for the inference engines themselves.
- Not a full Kubernetes platform.
- Not multi-node tensor-parallel inference across hosts.
- Not training or fine-tuning.
- Not adapter-first as a product identity; adapters remain a supported
  capability.

## Product Language

Use this language in product docs:

> serve-engine is a local-first inference router and lifecycle manager for
> teams running model-serving services on their own GPUs.

Avoid making LoRA/adapters the lead message. Prefer "services",
"routing", "lifecycle", "health", "policy", "backpressure", and
"observability".
