from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MemoryPressure(str, Enum):
    NORMAL   = "normal"
    ELEVATED = "elevated"
    HIGH     = "high"
    CRITICAL = "critical"


class MemoryMonitor:
    """
    Polls all workers every poll_interval_ms and aggregates memory usage.
    Emits pressure-level callbacks when thresholds are crossed.
    """

    def __init__(
        self,
        poll_interval_ms: int = 2000,
        elevated_threshold: float = 0.70,
        high_threshold: float = 0.80,
        critical_threshold: float = 0.90,
    ) -> None:
        self._interval = poll_interval_ms / 1000.0
        self._elevated = elevated_threshold
        self._high = high_threshold
        self._critical = critical_threshold

        self._backends: dict[str, object] = {}
        self._pressure: MemoryPressure = MemoryPressure.NORMAL
        self._callbacks: list[Callable[[MemoryPressure], None]] = []
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    def register_backend(self, worker_id: str, backend: object) -> None:
        self._backends[worker_id] = backend

    def deregister_backend(self, worker_id: str) -> None:
        self._backends.pop(worker_id, None)

    def on_pressure_change(self, callback: Callable[[MemoryPressure], None]) -> None:
        self._callbacks.append(callback)

    @property
    def current_pressure(self) -> MemoryPressure:
        return self._pressure

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="helix-memory-monitor")
        logger.info("MemoryMonitor started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll()
            except Exception as exc:
                logger.warning("MemoryMonitor poll error: %s", exc)
            await asyncio.sleep(self._interval)

    async def _poll(self) -> None:
        if not self._backends:
            return
        used_total = 0.0
        total_total = 0.0
        for backend in self._backends.values():
            try:
                stats = await backend.get_stats()  # type: ignore[attr-defined]
                used_total += stats.memory_used_gb
                total_total += stats.memory_total_gb
            except Exception:
                pass

        if total_total == 0:
            return

        utilization = used_total / total_total
        new_pressure = self._classify(utilization)
        if new_pressure != self._pressure:
            logger.info(
                "Memory pressure changed: %s → %s (utilization=%.1f%%)",
                self._pressure.value,
                new_pressure.value,
                utilization * 100,
            )
            self._pressure = new_pressure
            for cb in self._callbacks:
                try:
                    cb(new_pressure)
                except Exception as exc:
                    logger.error("Pressure callback error: %s", exc)

    def _classify(self, utilization: float) -> MemoryPressure:
        if utilization >= self._critical:
            return MemoryPressure.CRITICAL
        if utilization >= self._high:
            return MemoryPressure.HIGH
        if utilization >= self._elevated:
            return MemoryPressure.ELEVATED
        return MemoryPressure.NORMAL
