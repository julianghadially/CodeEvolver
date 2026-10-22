"""Dual authentication middleware: Clerk JWT + API key.

Both paths resolve to a clerk_id which is the universal user identifier.

Headers expected:
  Authorization: Bearer <token>
  Authorization-Type: clerk | api_key  (defaults to "clerk")
"""

import hashlib
import time

from fastapi import Request, HTTPException
from jose import jwt, JWTError
import httpx

from src.config import settings
from src.db import get_database

# JWKS cache with TTL
_jwks_cache: dict | None = None
_jwks_fetched_at: float = 0
_JWKS_TTL_SECONDS = 3600  # Re-fetch JWKS every hour


async def _get_jwks() -> dict:
    """Fetch and cache Clerk's JWKS for JWT verification."""
    global _jwks_cache, _jwks_fetched_at

    now = time.time()
    if _jwks_cache is not None and (now - _jwks_fetched_at) < _JWKS_TTL_SECONDS:
        return _jwks_cache

    if not settings.clerk_jwks_url:
        raise HTTPException(status_code=500, detail="Clerk JWKS URL not configured")

    async with httpx.AsyncClient() as client:
        resp = await client.get(settings.clerk_jwks_url, timeout=10.0)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        _jwks_fetched_at = now
        return _jwks_cache


async def _resolve_clerk_jwt(token: str) -> str:
    """Verify a Clerk JWT and return the clerk_id (sub claim)."""
    jwks = await _get_jwks()
    try:
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            issuer=settings.clerk_issuer,
            options={"verify_aud": False},
        )
        clerk_id = payload.get("sub")
        if not clerk_id:
            raise HTTPException(status_code=401, detail="JWT missing 'sub' claim")
        return clerk_id
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Clerk JWT: {e}")


async def _resolve_api_key(token: str) -> str:
    """Hash the API key and look up the clerk_id in MongoDB."""
    api_hash = hashlib.sha256(token.encode()).hexdigest()
    db = get_database()
    auth_doc = await db.auth.find_one({"api_hash": api_hash, "revoked": False})
    if not auth_doc:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return auth_doc["clerk_id"]


async def resolve_clerk_id(request: Request) -> str:
    """Extract clerk_id from either a Clerk JWT or an API key.

    Reads:
      Authorization: Bearer <token>
      Authorization-Type: clerk | api_key
    """
    auth_header = request.headers.get("Authorization", "")
    auth_type = request.headers.get("Authorization-Type", "clerk").lower()

    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = auth_header.removeprefix("Bearer ").strip()

    if auth_type == "api_key":
        return await _resolve_api_key(token)
    else:
        return await _resolve_clerk_jwt(token)
