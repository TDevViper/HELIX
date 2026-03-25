from __future__ import annotations

import asyncio
import logging
import random
from typing import Callable, Optional

from helix.control.hash_ring import ConsistentHashRing
from helix.models.worker import WorkerNode, WorkerStatus

logger = logging.getLogger(__name__)


class WorkerRegistry:
    """
    Maintains the live set of workers and runs gossip-based failure detection.

    Gossip protocol:
    - Every gossip_interval_ms, ping gossip_fan_out random peers
    - Miss = increment suspect_count on that worker
    - suspect_count >= threshold → DEAD, removed from hash ring
    - Successful ping resets suspect_count to 0

    This gives O(log N) convergence with no single point of failure.
    """

    def __init__(
        self,
        ring: ConsistentHashRing,
        gossip_interval_ms: int = 500,
        gossip_fan_out: int = 3,
        suspect_threshold: int = 3,
    ) -> None:
        self._ring = ring
        self._backends: dict[str, object] = {}
        self._gossip_interval = gossip_interval_ms / 1000.0
        self._fan_out = gossip_fan_out
        self._suspect_threshold = suspect_threshold

        # worker_id -> WorkerNode (source of truth)
        self._workers: dict[str, WorkerNode] = {}

        # Callbacks fired on worker death — scheduler uses this to requeue
        self._on_death_callbacks: list[Callable[[str], None]] = []

        self._gossip_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    # ── Registration ──────────────────────────────────────────

    def register(self, worker: WorkerNode) -> None:
        """
        Add a worker to the registry and hash ring.
        Safe to call multiple times — idempotent.
        """
        if worker.worker_id in self._workers:
            logger.info("Worker already registered: %s", worker.worker_id)
            return
        self._workers[worker.worker_id] = worker
        self._ring.add_node(worker)
        logger.info(
            "Registered worker %s (%s) at %s",
            worker.worker_id,
            worker.backend_type.value,
            worker.base_url,
        )

    def deregister(self, worker_id: str) -> None:
        """
        Remove a worker from the registry and hash ring.
        Called on graceful shutdown or confirmed death.
        """
        if worker_id not in self._workers:
            return
        self._workers.pop(worker_id)
        self._ring.remove_node(worker_id)
        logger.info("Deregistered worker %s", worker_id)

    def on_worker_death(self, callback: Callable[[str], None]) -> None:
        """Register a callback to fire when a worker is confirmed dead."""
        self._on_death_callbacks.append(callback)

    # ── Queries ───────────────────────────────────────────────

    def get_healthy_workers(self) -> list[WorkerNode]:
        return [
            w for w in self._workers.values()
            if w.status == WorkerStatus.HEALTHY
        ]

    def get_available_workers(self) -> list[WorkerNode]:
        """Healthy AND have capacity for another request."""
        return [w for w in self.get_healthy_workers() if w.is_available]

    def get_worker(self, worker_id: str) -> Optional[WorkerNode]:
        return self._workers.get(worker_id)

    def all_workers(self) -> list[WorkerNode]:
        return list(self._workers.values())

    def worker_count(self) -> int:
        return len(self._workers)

    # ── Gossip ────────────────────────────────────────────────

    async def start_gossip(self) -> None:
        """Start the background gossip loop as an asyncio task."""
        self._running = True
        self._gossip_task = asyncio.create_task(
            self._gossip_loop(), name="helix-gossip"
        )
        logger.info("Gossip loop started (interval=%.2fs)", self._gossip_interval)

    async def stop_gossip(self) -> None:
        """Gracefully stop the gossip loop."""
        self._running = False
        if self._gossip_task and not self._gossip_task.done():
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
        logger.info("Gossip loop stopped")

    async def _gossip_loop(self) -> None:
        """
        Main gossip loop — runs forever until stopped.
        Each tick: pick fan_out random workers, ping them,
        update suspect counts, fire death callbacks.
        """
        while self._running:
            try:
                await self._gossip_round()
            except Exception as exc:
                logger.warning("Gossip round error: %s", exc)
            await asyncio.sleep(self._gossip_interval)

    async def _gossip_round(self) -> None:
        """
        Single gossip round:
        1. Sample fan_out random workers
        2. Ping each via health_check
        3. On miss: increment suspect_count
        4. On hit: reset suspect_count, mark HEALTHY
        5. On confirmed dead: deregister + fire callbacks
        """
        workers = list(self._workers.values())
        if not workers:
            return

        sample_size = min(self._fan_out, len(workers))
        peers = random.sample(workers, sample_size)

        dead_ids: list[str] = []

        for peer in peers:
            alive = await self._ping(peer)
            if alive:
                peer.touch()
            else:
                just_died = peer.increment_suspect(self._suspect_threshold)
                if just_died:
                    dead_ids.append(peer.worker_id)
                    logger.warning(
                        "Worker confirmed dead: %s (suspect_count=%d)",
                        peer.worker_id,
                        peer.suspect_count,
                    )

        for worker_id in dead_ids:
            await self._on_worker_confirmed_dead(worker_id)

    async def _ping(self, worker: WorkerNode) -> bool:
        """
        Ping a worker by importing and calling its backend health_check.
        We store the backend instance on the WorkerNode via a side-channel
        dict keyed by worker_id. Returns False on any exception.
        """
        backend = self._backends.get(worker.worker_id)
        if backend is None:
            # No backend instance registered — treat as unhealthy
            return False
        try:
            return await asyncio.wait_for(backend.health_check(), timeout=2.0)
        except Exception:
            return False

    async def _on_worker_confirmed_dead(self, worker_id: str) -> None:
        """
        Called when gossip confirms a worker is dead:
        1. Remove from registry and ring
        2. Fire all registered death callbacks (scheduler requeues, etc.)
        """
        self.deregister(worker_id)
        for callback in self._on_death_callbacks:
            try:
                callback(worker_id)
            except Exception as exc:
                logger.error(
                    "Death callback error for %s: %s", worker_id, exc
                )

    # ── Backend registry (side-channel) ───────────────────────

    # Maps worker_id -> LLMBackend instance for gossip pings
    # Populated by WorkerPool when backends are started
    # _backends initialized as instance variable in __init__

    def register_backend(self, worker_id: str, backend: object) -> None:
        """Called by WorkerPool to give gossip access to health_check()."""
        self._backends[worker_id] = backend

    def deregister_backend(self, worker_id: str) -> None:
        self._backends.pop(worker_id, None)

    # ── Status summary ────────────────────────────────────────

    def status_summary(self) -> dict[str, object]:
        healthy = self.get_healthy_workers()
        return {
            "total_workers": len(self._workers),
            "healthy": len(healthy),
            "suspected": sum(
                1 for w in self._workers.values()
                if w.status == WorkerStatus.SUSPECTED
            ),
            "dead": sum(
                1 for w in self._workers.values()
                if w.status == WorkerStatus.DEAD
            ),
            "workers": [w.to_dict() for w in self._workers.values()],
        }
