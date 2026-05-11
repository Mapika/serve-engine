from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

BASE_URL = "http://daemon"


def _client(sock: Path) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(uds=str(sock))
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL, timeout=600.0)


async def get(sock: Path, path: str) -> Any:
    async with _client(sock) as c:
        r = await c.get(path)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"daemon error {r.status_code}: {detail}")
        return r.json()


async def post(sock: Path, path: str, *, json: dict[str, Any] | None = None) -> Any:
    async with _client(sock) as c:
        r = await c.post(path, json=json)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"daemon error {r.status_code}: {detail}")
        if r.status_code == 204:
            return None
        return r.json()


async def delete(sock: Path, path: str) -> None:
    async with _client(sock) as c:
        r = await c.delete(path)
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"daemon error {r.status_code}: {r.text}")
