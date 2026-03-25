from __future__ import annotations

from fastapi import HTTPException, Request, status

from helix.config import settings


async def verify_api_key(request: Request) -> str:
    """
    Middleware-style dependency. Returns client_id if auth passes.
    If auth_enabled=False in settings, always passes through.
    """
    if not settings.auth_enabled:
        return request.headers.get("X-Client-ID", "anonymous")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    token = auth_header.removeprefix("Bearer ").strip()
    if token not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return token
