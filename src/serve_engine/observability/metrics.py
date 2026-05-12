from __future__ import annotations

import asyncio

import httpx

from serve_engine.observability.trtllm_metrics import translate_many


def format_daemon_metrics(
    *,
    deployments_by_status: dict[str, int],
    models_total: int,
    api_keys_active: int,
    request_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# HELP serve_deployments Count of deployments by status.")
    lines.append("# TYPE serve_deployments gauge")
    for status, n in sorted(deployments_by_status.items()):
        lines.append(f'serve_deployments{{status="{status}"}} {n}')
    lines.append("# HELP serve_models_total Number of registered models.")
    lines.append("# TYPE serve_models_total gauge")
    lines.append(f"serve_models_total {models_total}")
    lines.append("# HELP serve_api_keys_active Number of non-revoked API keys.")
    lines.append("# TYPE serve_api_keys_active gauge")
    lines.append(f"serve_api_keys_active {api_keys_active}")
    lines.append("# HELP serve_proxy_requests_total Total /v1/* requests processed.")
    lines.append("# TYPE serve_proxy_requests_total counter")
    lines.append(f"serve_proxy_requests_total {request_count}")
    return "\n".join(lines) + "\n"


async def fetch_engine_metrics(base_url: str, path: str = "/metrics") -> str:
    """Best-effort fetch of an engine's metrics. Returns '' on failure.

    Timeout is 5s, not the typical Prometheus 2s, because TRT-LLM's /metrics
    on the PyTorch backend blocks waiting for the next iteration to complete
    and routinely takes 3-4s under light load. vLLM and SGLang respond in
    well under 100ms; the 5s ceiling only matters for slow engines and is
    well within Prometheus' default scrape_timeout.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(base_url.rstrip("/") + path)
            if r.status_code == 200:
                return r.text
    except httpx.HTTPError:
        pass
    return ""


def _looks_like_json(body: str) -> bool:
    """True if `body` is a JSON value (TRT-LLM emits an array, sometimes
    wrapped in whitespace). Prometheus exposition starts with `#` (HELP/TYPE)
    or a metric name char — never `[` or `{` — so a leading bracket/brace
    after stripping is an unambiguous signal. We avoid relying on the
    Content-Type header because TRT-LLM has historically returned
    `text/plain` for its JSON payload, making the body shape more reliable.
    """
    s = body.lstrip()
    return s.startswith(("[", "{"))


async def gather_engine_metrics(engine_urls: list[tuple[int, str]]) -> str:
    """engine_urls is [(deployment_id, base_url)].

    Prometheus-exposition bodies (vLLM, SGLang) pass through unchanged with
    a per-deployment header comment. JSON bodies (TRT-LLM) are batched and
    routed through the translator so that shared `# HELP` / `# TYPE` lines
    appear exactly once across all TRT-LLM deployments — required by strict
    Prometheus scrapers, which reject duplicate metadata for the same
    metric name.
    """
    if not engine_urls:
        return ""
    bodies = await asyncio.gather(
        *(fetch_engine_metrics(url) for _, url in engine_urls),
        return_exceptions=False,
    )
    out: list[str] = []
    json_bodies: list[tuple[int, str]] = []
    for (dep_id, _), body in zip(engine_urls, bodies, strict=True):
        if not body:
            continue
        if _looks_like_json(body):
            json_bodies.append((dep_id, body))
            continue
        out.append(f"# --- deployment {dep_id} ---")
        out.append(body.rstrip())
    if json_bodies:
        translated = translate_many(json_bodies).rstrip()
        if translated:
            out.append("# --- trtllm deployments ---")
            out.append(translated)
    return "\n".join(out) + ("\n" if out else "")
