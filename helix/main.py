from __future__ import annotations

import logging

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI

from helix.cache.lru_eviction import GlobalLRUEviction
from helix.cache.prefix_cache import DistributedPrefixCache
from helix.config import settings
from helix.control.hash_ring import ConsistentHashRing
from helix.control.memory_monitor import MemoryMonitor
from helix.control.node_registry import WorkerRegistry
from helix.control.scheduler import PreemptiveScheduler
from helix.data.kv_coordinator import KVCacheCoordinator
from helix.data.worker_pool import WorkerPool
from helix.gateway.router import router
from helix.observability.health import health_router
from helix.observability.tracer import setup_tracing

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Helix", description="Distributed LLM Inference Scheduler", version="1.0.0")
    app.include_router(router)
    app.include_router(health_router)

    @app.on_event("startup")
    async def startup() -> None:
        setup_tracing("helix")

        redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
        prefix_cache = DistributedPrefixCache(redis_client)
        lru = GlobalLRUEviction(redis_client)
        kv_coordinator = KVCacheCoordinator(prefix_cache)

        ring = ConsistentHashRing(vnodes=150)
        registry = WorkerRegistry(
            ring=ring,
            gossip_interval_ms=settings.gossip_interval_ms,
            gossip_fan_out=settings.gossip_fan_out,
            suspect_threshold=settings.gossip_suspect_threshold,
        )
        memory_monitor = MemoryMonitor(
            elevated_threshold=settings.memory_elevated_threshold,
            high_threshold=settings.memory_high_threshold,
            critical_threshold=settings.memory_critical_threshold,
        )
        scheduler = PreemptiveScheduler(
            memory_monitor=memory_monitor,
            poll_interval_ms=settings.scheduler_poll_interval_ms,
        )

        pool = WorkerPool(ring=ring, registry=registry)
        scheduler.set_pool(pool)

        registry.on_worker_death(
            lambda wid: __import__("asyncio").create_task(kv_coordinator.on_worker_death(wid))
        )

        await pool.startup()

        for wid, backend in pool._backends.items():
            memory_monitor.register_backend(wid, backend)

        await registry.start_gossip()
        await memory_monitor.start()
        await scheduler.start()

        app.state.redis = redis_client
        app.state.ring = ring
        app.state.registry = registry
        app.state.worker_pool = pool
        app.state.scheduler = scheduler
        app.state.memory_monitor = memory_monitor
        app.state.kv_coordinator = kv_coordinator
        app.state.lru = lru

        logger.info("Helix started — workers: %d", registry.worker_count())

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await app.state.scheduler.stop()
        await app.state.memory_monitor.stop()
        await app.state.registry.stop_gossip()
        await app.state.worker_pool.shutdown()
        await app.state.redis.aclose()
        logger.info("Helix shutdown complete")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("helix.main:app", host=settings.host, port=settings.port, log_level=settings.log_level, reload=False)
