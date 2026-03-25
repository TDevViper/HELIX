from __future__ import annotations

import logging
import time
from typing import List

logger = logging.getLogger(__name__)

_LRU_KEY = "helix:prefix_lru"


class GlobalLRUEviction:
    """
    Tracks prefix cache access times in a Redis sorted set.
    Score = Unix timestamp of last access → lowest score = oldest = evict first.
    """

    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client

    async def on_cache_hit(self, prefix_hash: str) -> None:
        """Refresh score to now on every hit."""
        try:
            await self._redis.zadd(_LRU_KEY, {prefix_hash: time.time()})
        except Exception as exc:
            logger.warning("LRU update error: %s", exc)

    async def get_eviction_candidates(self, n: int = 10) -> List[str]:
        """Return n oldest prefix hashes (lowest scores = least recently used)."""
        try:
            results = await self._redis.zrange(_LRU_KEY, 0, n - 1)
            return [r.decode() if isinstance(r, bytes) else r for r in results]
        except Exception as exc:
            logger.warning("LRU candidates error: %s", exc)
            return []

    async def evict_lru(self, count: int = 10) -> int:
        """Evict the `count` least recently used entries. Returns actual count evicted."""
        try:
            candidates = await self.get_eviction_candidates(count)
            if not candidates:
                return 0
            pipe = self._redis.pipeline()
            pipe.hdel("helix:prefix_cache", *candidates)
            pipe.zrem(_LRU_KEY, *candidates)
            await pipe.execute()
            logger.info("LRU evicted %d entries", len(candidates))
            return len(candidates)
        except Exception as exc:
            logger.warning("LRU evict error: %s", exc)
            return 0
