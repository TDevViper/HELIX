from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from helix.control.hash_ring import ConsistentHashRing
from helix.control.node_registry import WorkerRegistry
from helix.data.worker_pool import WorkerPool
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerNode


def _make_worker(wid: str) -> WorkerNode:
    return WorkerNode(worker_id=wid, base_url=f"http://{wid}", backend_type=BackendType.OLLAMA)


async def test_dead_worker_removed_from_ring():
    ring = ConsistentHashRing(vnodes=50)
    registry = WorkerRegistry(ring=ring)

    registry.register(_make_worker("w1"))
    registry.register(_make_worker("w2"))
    assert ring.get_node("key") in {"w1", "w2"}

    registry.deregister("w1")
    assert ring.get_node("key") == "w2"


async def test_pool_raises_when_backend_missing():
    """Pool raises RuntimeError when ring resolves a worker but backend dict lacks it."""
    ring = ConsistentHashRing(vnodes=50)
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)

    # Register worker in ring but do NOT add a backend instance
    registry.register(_make_worker("w1"))

    req = HelixRequest(model="llama3", messages=[])
    with pytest.raises(RuntimeError, match="Backend instance missing"):
        async for _ in pool.generate(req):
            pass


async def test_kv_cache_invalidated_on_worker_death():
    from helix.cache.prefix_cache import DistributedPrefixCache
    from helix.data.kv_coordinator import KVCacheCoordinator

    redis = MagicMock()
    redis.hgetall = AsyncMock(return_value={
        b"hash1": b"dead-worker",
        b"hash2": b"live-worker",
    })
    pipe = MagicMock()
    pipe.hdel = MagicMock()
    pipe.zrem = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, 1])
    redis.pipeline = MagicMock(return_value=pipe)

    cache = DistributedPrefixCache(redis)
    coordinator = KVCacheCoordinator(cache)

    await coordinator.on_worker_death("dead-worker")
    pipe.execute.assert_called_once()
