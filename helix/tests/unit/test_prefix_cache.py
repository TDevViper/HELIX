from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from helix.cache.prefix_cache import DistributedPrefixCache, hash_prefix


def test_hash_prefix_empty_messages():
    # Single message → no prefix → empty string
    assert hash_prefix([{"role": "user", "content": "hi"}]) == ""


def test_hash_prefix_deterministic():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "follow-up"},
    ]
    assert hash_prefix(msgs) == hash_prefix(msgs)


def test_hash_prefix_changes_with_content():
    msgs_a = [{"role": "system", "content": "A"}, {"role": "user", "content": "q"}]
    msgs_b = [{"role": "system", "content": "B"}, {"role": "user", "content": "q"}]
    assert hash_prefix(msgs_a) != hash_prefix(msgs_b)


def _make_cache() -> tuple[DistributedPrefixCache, MagicMock]:
    redis = MagicMock()
    redis.hget = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    pipe = MagicMock()
    pipe.hset = MagicMock()
    pipe.zadd = MagicMock()
    pipe.hdel = MagicMock()
    pipe.zrem = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, 1])
    redis.pipeline = MagicMock(return_value=pipe)
    return DistributedPrefixCache(redis), redis


async def test_lookup_miss_returns_none():
    cache, _ = _make_cache()
    result = await cache.lookup("deadbeef")
    assert result is None


async def test_lookup_hit_returns_worker_id():
    cache, redis = _make_cache()
    redis.hget = AsyncMock(return_value=b"worker-1")
    result = await cache.lookup("deadbeef")
    assert result == "worker-1"


async def test_register_calls_pipeline():
    cache, redis = _make_cache()
    pipe = redis.pipeline()
    await cache.register("abc123", "worker-1")
    pipe.hset.assert_called_once_with("helix:prefix_cache", "abc123", "worker-1")
    pipe.execute.assert_called_once()


async def test_lookup_empty_hash_returns_none():
    cache, _ = _make_cache()
    assert await cache.lookup("") is None


async def test_invalidate_removes_entries():
    cache, redis = _make_cache()
    redis.hgetall = AsyncMock(return_value={
        b"hash1": b"worker-dead",
        b"hash2": b"worker-alive",
        b"hash3": b"worker-dead",
    })
    pipe = redis.pipeline()
    count = await cache.invalidate("worker-dead")
    assert count == 2
    pipe.execute.assert_called_once()
