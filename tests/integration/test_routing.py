import pytest
from unittest.mock import AsyncMock, MagicMock
from helix.control.hash_ring import ConsistentHashRing
from helix.control.node_registry import WorkerRegistry
from helix.data.worker_pool import WorkerPool
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerNode


def _make_worker(wid: str, url: str = "http://localhost:11434") -> WorkerNode:
    return WorkerNode(worker_id=wid, base_url=url, backend_type=BackendType.OLLAMA)


@pytest.mark.asyncio
async def test_request_routes_to_healthy_worker():
    ring = ConsistentHashRing(vnodes=50)
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)

    worker = _make_worker("w1")
    registry.register(worker)

    mock_backend = MagicMock()
    async def fake_generate(req):
        yield "hello"
    mock_backend.generate = fake_generate
    mock_backend.startup = AsyncMock()
    pool._backends["w1"] = mock_backend

    req = HelixRequest(model="llama3", messages=[])
    tokens = []
    async for tok in pool.generate(req):
        tokens.append(tok)
    assert tokens == ["hello"]


@pytest.mark.asyncio
async def test_no_workers_raises():
    ring = ConsistentHashRing()
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)
    req = HelixRequest(model="llama3", messages=[])
    with pytest.raises(RuntimeError, match="No available worker"):
        async for _ in pool.generate(req):
            pass
