from __future__ import annotations

from fastapi import APIRouter, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

health_router = APIRouter()


@health_router.get("/v1/stats")
async def stats(request: Request) -> dict:
    pool = request.app.state.worker_pool
    scheduler = request.app.state.scheduler
    monitor = request.app.state.memory_monitor
    return {
        "pool": pool.pool_stats() if pool else {},
        "scheduler": {
            "queue_depth": scheduler.queue_depth() if scheduler else 0,
            "in_flight": scheduler.in_flight_count() if scheduler else 0,
        },
        "memory_pressure": monitor.current_pressure.value if monitor else "unknown",
    }


@health_router.get("/metrics")
async def prometheus_metrics() -> bytes:
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
