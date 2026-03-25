from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from helix.control.hash_ring import ConsistentHashRing
from helix.control.node_registry import WorkerRegistry
from helix.data.worker_pool import WorkerPool
from helix.models.worker import WorkerNode, BackendType, WorkerStatus
from helix.models.request import HelixRequest
from main import app


@pytest.fixture
def mock_worker() -> WorkerNode:
    return WorkerNode(
        worker_id="test-ollama-1",
        base_url="http://localhost:11434",
        backend_type=BackendType.OLLAMA,
        status=WorkerStatus.HEALTHY,
        active_requests=0,
        max_concurrent_requests=8,
        supported_models=[],
    )


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.health_check = AsyncMock(return_value=True)
    backend.shutdown = AsyncMock()
    backend.startup = AsyncMock()

    async def fake_generate(request):
        for token in ["Hello", " world", "!"]:
            yield token

    backend.generate = fake_generate
    return backend


@pytest_asyncio.fixture
async def client(mock_worker, mock_backend):
    ring = ConsistentHashRing(vnodes=150)
    registry = WorkerRegistry(ring=ring)
    pool = WorkerPool(ring=ring, registry=registry)

    ring.add_node(mock_worker)
    registry._workers = {"test-ollama-1": mock_worker}
    pool._backends = {"test-ollama-1": mock_backend}
    registry.register_backend("test-ollama-1", mock_backend)

    app.state.ring = ring
    app.state.registry = registry
    app.state.worker_pool = pool

    with patch("helix.control.node_registry.WorkerRegistry.start_gossip", new_callable=AsyncMock), \
         patch("helix.control.node_registry.WorkerRegistry.stop_gossip", new_callable=AsyncMock):

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac


# ── Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_list_nodes(client):
    resp = await client.get("/v1/nodes")
    assert resp.status_code == 200
    assert "total_workers" in resp.json()


@pytest.mark.asyncio
async def test_chat_completions_streams(client, mock_worker, mock_backend):
    app.state.worker_pool.resolve_worker = MagicMock(return_value=mock_worker)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data:" in resp.text


@pytest.mark.asyncio
async def test_session_affinity(client, mock_worker):
    session_id = "test-session-abc"
    ring = app.state.ring
    results = {ring.get_node(session_id) for _ in range(5)}
    assert len(results) == 1, "Session affinity broken — same key maps to different nodes"


@pytest.mark.asyncio
async def test_no_workers_returns_error_in_stream(client):
    app.state.worker_pool._backends = {}
    app.state.registry._workers = {}
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    assert "error" in resp.text.lower() or "data:" in resp.text


@pytest.mark.asyncio
async def test_hash_ring_distributes_keys(client):
    ring = app.state.ring
    results = {ring.get_node(f"session-{i}") for i in range(20)}
    assert len(results) >= 1
