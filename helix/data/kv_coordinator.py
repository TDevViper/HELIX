from __future__ import annotations

import logging
from typing import Optional

from helix.cache.prefix_cache import DistributedPrefixCache

logger = logging.getLogger(__name__)


class KVCacheCoordinator:
    """
    Coordinates KV cache state across workers.
    On worker death: invalidates stale entries and logs reassignment.
    """

    def __init__(self, prefix_cache: DistributedPrefixCache) -> None:
        self._cache = prefix_cache
        self._hit_count = 0
        self._miss_count = 0

    async def on_worker_death(self, worker_id: str) -> None:
        """
        Called by WorkerRegistry death callback.
        Invalidates all prefix cache entries owned by the dead worker.
        """
        count = await self._cache.invalidate(worker_id)
        logger.warning(
            "KVCoordinator: worker %s died, invalidated %d cache entries",
            worker_id, count,
        )

    async def check_prefix(self, prefix_hash: str) -> Optional[str]:
        """
        Returns owning worker_id on cache hit, None on miss.
        Updates hit/miss counters for metrics.
        """
        if not prefix_hash:
            self._miss_count += 1
            return None
        result = await self._cache.lookup(prefix_hash)
        if result:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result

    async def record_prefix(self, prefix_hash: str, worker_id: str) -> None:
        await self._cache.register(prefix_hash, worker_id)

    def get_cache_hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total
