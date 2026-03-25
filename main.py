from __future__ import annotations

import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from helix.control.hash_ring import ConsistentHashRing
from helix.control.node_registry import WorkerRegistry
from helix.data.worker_pool import WorkerPool
from helix.gateway.router import router

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("helix.main")

# ── App factory ───────────────────────────────────────────────
app = FastAPI(
    title="Helix",
    description="Distributed LLM Inference Scheduler",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# ── App state (shared across requests) ────────────────────────
app.state.ring = None
app.state.registry = None
app.state.worker_pool = None


# ── Lifespan ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting Helix v0.1.0")

    # 1. Build hash ring
    ring = ConsistentHashRing(vnodes=settings.virtual_nodes)
    app.state.ring = ring
    logger.info("Hash ring initialized (vnodes=%d)", settings.virtual_nodes)

    # 2. Build registry + wire gossip
    registry = WorkerRegistry(
        ring=ring,
        gossip_interval_ms=settings.gossip_interval_ms,
        gossip_fan_out=settings.gossip_fan_out,
        suspect_threshold=settings.gossip_suspect_threshold,
    )
    app.state.registry = registry

    # 3. Build worker pool + bootstrap workers from env
    pool = WorkerPool(ring=ring, registry=registry)
    await pool.startup()
    app.state.worker_pool = pool
    logger.info(
        "Worker pool ready — %d workers registered",
        registry.worker_count(),
    )

    # 4. Start gossip loop
    await registry.start_gossip()
    logger.info("Gossip loop running")

    logger.info(
        "Helix listening on %s:%d",
        settings.host,
        settings.port,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("Shutting down Helix...")

    registry: WorkerRegistry | None = app.state.registry
    pool: WorkerPool | None = app.state.worker_pool

    if registry:
        await registry.stop_gossip()

    if pool:
        await pool.shutdown()

    logger.info("Helix shutdown complete")


# ── Entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )
