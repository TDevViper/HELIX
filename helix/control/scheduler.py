from __future__ import annotations

import asyncio
import heapq
import logging
from typing import Optional

from helix.control.memory_monitor import MemoryMonitor, MemoryPressure
from helix.models.request import HelixRequest

logger = logging.getLogger(__name__)


class PreemptiveScheduler:
    """
    Priority queue scheduler with memory-pressure-aware preemption.
    - Higher priority_score = dispatched first
    - Under CRITICAL pressure: lowest-priority in-flight request is preempted
    - Preempted requests are re-queued with is_resumed=True
    """

    def __init__(
        self,
        memory_monitor: MemoryMonitor,
        poll_interval_ms: int = 50,
    ) -> None:
        self._monitor = memory_monitor
        self._interval = poll_interval_ms / 1000.0
        # Min-heap: (neg_priority_score, arrived_at, request)
        self._queue: list[tuple[float, float, HelixRequest]] = []
        # request_id -> HelixRequest for in-flight requests
        self._in_flight: dict[str, HelixRequest] = {}
        self._lock = asyncio.Lock()
        self._dispatch_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False
        # Injected by main.py after pool is ready
        self._pool: Optional[object] = None

        memory_monitor.on_pressure_change(self._on_pressure_change)

    def set_pool(self, pool: object) -> None:
        self._pool = pool

    async def enqueue(self, request: HelixRequest) -> None:
        async with self._lock:
            score = request.priority_score()
            heapq.heappush(self._queue, (-score, request.arrived_at, request))
            self._dispatch_event.set()
            logger.debug(
                "Enqueued request %s (score=%.6f, queue_depth=%d)",
                request.request_id, score, len(self._queue),
            )

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop(), name="helix-scheduler")
        logger.info("Scheduler started")

    async def stop(self) -> None:
        self._running = False
        self._dispatch_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _dispatch_loop(self) -> None:
        while self._running:
            await asyncio.wait_for(
                self._dispatch_event.wait(), timeout=self._interval
            ) if False else None
            self._dispatch_event.clear()

            async with self._lock:
                expired: list[HelixRequest] = []
                ready: list[HelixRequest] = []
                temp: list[tuple[float, float, HelixRequest]] = []

                while self._queue:
                    neg_score, arrived, req = heapq.heappop(self._queue)
                    if req.is_expired():
                        expired.append(req)
                    else:
                        ready.append(req)
                        temp.append((neg_score, arrived, req))

                for req in expired:
                    req.mark_failed()
                    logger.warning("Expired request dropped: %s", req.request_id)

                # Re-push non-expired back
                for item in temp:
                    heapq.heappush(self._queue, item)

                # Dispatch one request if pool is ready
                if self._queue and self._pool is not None:
                    _, _, req = heapq.heappop(self._queue)
                    self._in_flight[req.request_id] = req
                    asyncio.create_task(self._dispatch(req))

            await asyncio.sleep(self._interval)

    async def _dispatch(self, request: HelixRequest) -> None:
        try:
            async for _ in self._pool.generate(request):  # type: ignore[attr-defined]
                pass
        except Exception as exc:
            logger.error("Dispatch error for %s: %s", request.request_id, exc)
            request.mark_failed()
        finally:
            self._in_flight.pop(request.request_id, None)

    def _on_pressure_change(self, pressure: MemoryPressure) -> None:
        if pressure == MemoryPressure.CRITICAL:
            asyncio.create_task(self._preempt_lowest_priority())

    async def _preempt_lowest_priority(self) -> None:
        if not self._in_flight:
            return
        # Find lowest priority_score in-flight
        victim = min(self._in_flight.values(), key=lambda r: r.priority_score())
        checkpoint_key = f"helix:checkpoint:{victim.request_id}"
        victim.mark_preempted(checkpoint_key)
        self._in_flight.pop(victim.request_id, None)
        # Re-queue for resume
        await self.enqueue(victim)
        logger.warning(
            "Preempted request %s under CRITICAL pressure (checkpoint=%s)",
            victim.request_id, checkpoint_key,
        )

    def queue_depth(self) -> int:
        return len(self._queue)

    def in_flight_count(self) -> int:
        return len(self._in_flight)
