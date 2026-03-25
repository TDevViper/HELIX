from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def hash_prefix(messages: list[dict]) -> str:
    prefix = messages[:-1]
    if not prefix:
        return ""
    canonical = json.dumps(prefix, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class DistributedPrefixCache:
    _HASH_KEY = "helix:prefix_cache"
    _LRU_KEY = "helix:prefix_lru"

    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client

    async def lookup(self, prefix_hash: str) -> Optional[str]:
        if not prefix_hash:
            return None
        try:
            result = await self._redis.hget(self._HASH_KEY, prefix_hash)
            return result.decode() if result else None
        except Exception as exc:
            logger.warning("Prefix cache lookup error: %s", exc)
            return None

    async def register(self, prefix_hash: str, worker_id: str, token_count: int = 0) -> None:
        if not prefix_hash:
            return
        try:
            pipe = self._redis.pipeline()
            pipe.hset(self._HASH_KEY, prefix_hash, worker_id)
            pipe.zadd(self._LRU_KEY, {prefix_hash: time.time()})
            await pipe.execute()
        except Exception as exc:
            logger.warning("Prefix cache register error: %s", exc)

    async def invalidate(self, worker_id: str) -> int:
        try:
            all_entries = await self._redis.hgetall(self._HASH_KEY)
            to_remove = [k for k, v in all_entries.items() if v.decode() == worker_id]
            if to_remove:
                pipe = self._redis.pipeline()
                pipe.hdel(self._HASH_KEY, *to_remove)
                pipe.zrem(self._LRU_KEY, *to_remove)
                await pipe.execute()
            logger.info("Invalidated %d prefix cache entries for dead worker %s", len(to_remove), worker_id)
            return len(to_remove)
        except Exception as exc:
            logger.warning("Prefix cache invalidate error: %s", exc)
            return 0
