from __future__ import annotations

import asyncio

import httpx


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
    """Best-effort fetch of an engine's Prometheus metrics. Returns '' on failure."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(base_url.rstrip("/") + path)
            if r.status_code == 200:
                return r.text
    except httpx.HTTPError:
        pass
    return ""


async def gather_engine_metrics(engine_urls: list[tuple[int, str]]) -> str:
    """engine_urls is [(deployment_id, base_url)]. Concatenates with a header per dep."""
    if not engine_urls:
        return ""
    bodies = await asyncio.gather(
        *(fetch_engine_metrics(url) for _, url in engine_urls),
        return_exceptions=False,
    )
    out: list[str] = []
    for (dep_id, _), body in zip(engine_urls, bodies, strict=True):
        if not body:
            continue
        out.append(f"# --- deployment {dep_id} ---")
        out.append(body.rstrip())
    return "\n".join(out) + ("\n" if out else "")
