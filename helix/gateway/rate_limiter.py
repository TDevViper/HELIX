from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import HTTPException, Request, status


class TokenBucketRateLimiter:
    """
    Sliding window rate limiter — tracks request timestamps per client_id.
    Thread-safe for single-process asyncio usage (no locks needed).
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self._rpm = requests_per_minute
        self._window = 60.0
        # client_id -> deque of timestamps
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> bool:
        from helix.config import settings
        if not settings.rate_limit_enabled:
            return True

        now = time.time()
        bucket = self._buckets[client_id]
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self._rpm:
            return False
        bucket.append(now)
        return True


_limiter = TokenBucketRateLimiter()


async def rate_limit(request: Request) -> None:
    from helix.config import settings
    if not settings.rate_limit_enabled:
        return
    client_id = request.headers.get("X-Client-ID", request.client.host if request.client else "unknown")
    if not _limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
