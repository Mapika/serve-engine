from __future__ import annotations

import sqlite3

from fastapi import HTTPException, Request, status

from serve_engine.auth import limiter
from serve_engine.auth.tiers import Limits
from serve_engine.store import api_keys


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def require_auth_dep(request: Request) -> api_keys.ApiKey | None:
    """FastAPI dependency. Returns the ApiKey on success, raises 401/429 on failure.

    If no keys exist in the table, auth is bypassed (returns None).
    """
    conn: sqlite3.Connection = request.app.state.conn
    if api_keys.count_active(conn) == 0:
        return None

    auth_header = request.headers.get("authorization")
    secret = _extract_bearer(auth_header)
    if secret is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header (expected: Bearer sk-...)",
            headers={"WWW-Authenticate": 'Bearer realm="serve-engine"'},
        )

    key = api_keys.verify(conn, secret)
    if key is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="invalid or revoked API key",
        )

    tier_cfg: dict[str, Limits] = request.app.state.tier_cfg
    decision = limiter.check(conn, key=key, tier_cfg=tier_cfg)
    if isinstance(decision, limiter.Denied):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"{decision.limit_name} limit reached "
                f"({decision.current}/{decision.limit_value} in {decision.window_s}s)"
            ),
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    return key
