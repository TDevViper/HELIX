from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from helix.control.hash_ring import ConsistentHashRing
from helix.control.node_registry import WorkerRegistry
from helix.data.worker_pool import WorkerPool
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerNode


def _make_worker(wid: str, url: str = "http://localhost:11434") -> WorkerNode:
    return WorkerNode(worker_id=wid, base_url=url, backend_type=BackendType.OLLAMA)


async def test_request_routes_to_healthy_worker():
    ring = ConsistentHashRing(vnodes=50)
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)

    worker = _make_worker("w1")
    registry.register(worker)

    async def fake_generate(req):
        yield "hello"

    mock_backend = MagicMock()
    mock_backend.generate = fake_generate
    mock_backend.startup = AsyncMock()
    pool._backends["w1"] = mock_backend

    req = HelixRequest(model="llama3", messages=[])
    tokens = []
    async for tok in pool.generate(req):
        tokens.append(tok)
    assert tokens == ["hello"]


async def test_no_workers_raises():
    ring = ConsistentHashRing()
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)
    req = HelixRequest(model="llama3", messages=[])
    with pytest.raises(RuntimeError, match="No available worker"):
        async for _ in pool.generate(req):
            pass


async def test_session_affinity_routes_consistently():
    ring = ConsistentHashRing(vnodes=100)
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)

    for wid in ["w1", "w2", "w3"]:
        registry.register(_make_worker(wid))

    async def fake_generate(req):
        yield "tok"

    for wid in ["w1", "w2", "w3"]:
        m = MagicMock()
        m.generate = fake_generate
        m.startup = AsyncMock()
        pool._backends[wid] = m

    req = HelixRequest(model="llama3", messages=[], session_id="sess-42")
    first = ring.get_node("sess-42")
    assert ring.get_node("sess-42") == first  # deterministic
