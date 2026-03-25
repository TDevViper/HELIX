from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from helix.control.hash_ring import ConsistentHashRing
from helix.control.memory_monitor import MemoryMonitor
from helix.control.node_registry import WorkerRegistry
from helix.control.scheduler import PreemptiveScheduler
from helix.data.worker_pool import WorkerPool
from helix.main import create_app
from helix.models.worker import BackendType, WorkerNode


def _make_worker(wid: str) -> WorkerNode:
    return WorkerNode(worker_id=wid, base_url=f"http://{wid}", backend_type=BackendType.OLLAMA)


async def fake_generate(req):
    yield "hello"
    yield " world"


@pytest.fixture
async def app():
    """
    Creates a fully wired FastAPI app with mocked backends.
    Bypasses Redis and real workers entirely.
    """
    application = create_app()

    ring = ConsistentHashRing(vnodes=50)
    registry = WorkerRegistry(ring=ring)
    monitor = MemoryMonitor()
    scheduler = PreemptiveScheduler(memory_monitor=monitor)

    registry.register(_make_worker("w1"))

    mock_backend = MagicMock()
    mock_backend.generate = fake_generate
    mock_backend.startup = AsyncMock()

    pool = WorkerPool(ring=ring, registry=registry)
    pool._backends["w1"] = mock_backend
    scheduler.set_pool(pool)

    # Inject state directly — skip startup event
    application.state.registry = registry
    application.state.worker_pool = pool
    application.state.scheduler = scheduler
    application.state.memory_monitor = monitor
    application.state.redis = MagicMock()
    application.state.kv_coordinator = MagicMock()
    application.state.lru = MagicMock()

    return application


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_stats_shape(client):
    resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "scheduler" in body
    assert "queue_depth" in body["scheduler"]
    assert "in_flight" in body["scheduler"]
    assert "memory_pressure" in body


async def test_metrics_endpoint(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"helix_" in resp.content or b"python_" in resp.content


# ── Chat completions ──────────────────────────────────────────────────────────

async def test_chat_completions_streams_sse(client):
    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert b"data:" in resp.content
    assert b"[DONE]" in resp.content


async def test_chat_completions_missing_model_returns_422(client):
    # messages field must be a list of dicts with role/content — bad type triggers 422
    resp = await client.post("/v1/chat/completions", json={"model": "llama3", "messages": "bad"})
    assert resp.status_code == 422


async def test_chat_completions_empty_messages_accepted(client):
    payload = {"model": "llama3", "messages": []}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


# ── Auth ─────────────────────────────────────────────────────────────────────

async def test_auth_disabled_allows_any_request(client):
    """Default: auth_enabled=False so no key needed."""
    payload = {"model": "llama3", "messages": []}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


async def test_auth_enabled_rejects_missing_key(app):
    with patch("helix.gateway.auth.settings") as mock_settings:
        mock_settings.auth_enabled = True
        mock_settings.api_keys = ["valid-key-123"]
        mock_settings.rate_limit_enabled = False
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"model": "llama3", "messages": []},
            )
            assert resp.status_code in {401, 403}


async def test_auth_enabled_accepts_valid_key(app):
    with patch("helix.gateway.auth.settings") as mock_settings:
        mock_settings.auth_enabled = True
        mock_settings.api_keys = ["valid-key-123"]
        mock_settings.rate_limit_enabled = False
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"model": "llama3", "messages": []},
                headers={"Authorization": "Bearer valid-key-123"},
            )
            assert resp.status_code == 200


# ── Rate limiting ─────────────────────────────────────────────────────────────

async def test_rate_limit_triggers_429(app):
    with patch("helix.gateway.rate_limiter._limiter") as mock_limiter:
        mock_limiter.is_allowed.return_value = False
        with patch("helix.gateway.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.auth_enabled = False
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post(
                    "/v1/chat/completions",
                    json={"model": "llama3", "messages": []},
                )
                assert resp.status_code == 429
