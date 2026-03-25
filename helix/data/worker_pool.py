from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from helix.config import settings
from helix.control.hash_ring import ConsistentHashRing
from helix.control.node_registry import WorkerRegistry
from helix.data.backend_abc import LLMBackend
from helix.data.mlx_backend import MLXBackend
from helix.data.ollama_backend import OllamaBackend
from helix.data.vllm_backend import VLLMBackend
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerNode

logger = logging.getLogger(__name__)


def _make_backend(worker: WorkerNode) -> LLMBackend:
    """
    Factory: instantiate the correct LLMBackend subclass
    based on the worker's backend_type.
    """
    if worker.backend_type == BackendType.OLLAMA:
        return OllamaBackend(
            worker_id=worker.worker_id,
            base_url=worker.base_url,
            timeout_seconds=settings.worker_request_timeout_seconds,
        )
    elif worker.backend_type == BackendType.MLX:
        return MLXBackend(
            worker_id=worker.worker_id,
            base_url=worker.base_url,
            timeout_seconds=settings.worker_request_timeout_seconds,
        )
    elif worker.backend_type == BackendType.VLLM:
        return VLLMBackend(
            worker_id=worker.worker_id,
            base_url=worker.base_url,
            timeout_seconds=settings.worker_request_timeout_seconds,
        )
    else:
        raise ValueError(f"Unknown backend type: {worker.backend_type}")


class WorkerPool:
    """
    Manages the lifecycle of backend instances and routes
    requests to the correct worker via the hash ring.

    Responsibilities:
    - Bootstrap workers from config on startup
    - Start/stop backend HTTP clients
    - Route requests: hash ring lookup → backend.generate()
    - Track active request counts per worker
    - Expose pool-level stats for the memory monitor
    """

    def __init__(
        self,
        ring: ConsistentHashRing,
        registry: WorkerRegistry,
    ) -> None:
        self._ring = ring
        self._registry = registry
        # worker_id -> LLMBackend instance
        self._backends: dict[str, LLMBackend] = {}

    # ── Lifecycle ─────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Bootstrap workers from HELIX_BOOTSTRAP_WORKERS env var.
        Each URL is assumed to be an Ollama instance unless the URL
        contains a hint (mlx, vllm) in the hostname or port.
        """
        for url in settings.bootstrap_workers:
            backend_type = self._infer_backend_type(url)
            worker_id = f"{backend_type.value}-{url.split('://')[-1].replace('/', '-')}"
            worker = WorkerNode(
                worker_id=worker_id,
                base_url=url,
                backend_type=backend_type,
            )
            await self.add_worker(worker)
            logger.info(
                "Bootstrapped worker %s (%s)", worker_id, backend_type.value
            )

    async def shutdown(self) -> None:
        """Gracefully close all backend HTTP clients."""
        for worker_id, backend in list(self._backends.items()):
            try:
                await backend.shutdown()
                logger.info("Shut down backend %s", worker_id)
            except Exception as exc:
                logger.warning("Error shutting down %s: %s", worker_id, exc)
        self._backends.clear()

    async def add_worker(self, worker: WorkerNode) -> None:
        """
        Register a worker and start its backend client.
        Idempotent — safe to call if worker already exists.
        """
        if worker.worker_id in self._backends:
            logger.info("Worker already in pool: %s", worker.worker_id)
            return
        backend = _make_backend(worker)
        await backend.startup()
        self._backends[worker.worker_id] = backend
        self._registry.register(worker)
        self._registry.register_backend(worker.worker_id, backend)
        logger.info("Added worker to pool: %s", worker.worker_id)

    async def remove_worker(self, worker_id: str) -> None:
        """
        Deregister a worker and shut down its backend client.
        Called by gossip death callbacks.
        """
        backend = self._backends.pop(worker_id, None)
        if backend:
            try:
                await backend.shutdown()
            except Exception as exc:
                logger.warning(
                    "Error shutting down dead worker %s: %s", worker_id, exc
                )
        self._registry.deregister(worker_id)
        self._registry.deregister_backend(worker_id)
        logger.info("Removed worker from pool: %s", worker_id)

    # ── Routing ───────────────────────────────────────────────

    def resolve_worker(self, request: HelixRequest) -> Optional[WorkerNode]:
        """
        Consistent hash lookup: session_id → worker.
        Falls back to any available worker if the hashed worker is
        unavailable (dead or overloaded).
        """
        # Primary: consistent hash on session_id for cache affinity
        primary_id = self._ring.get_node(request.session_id)
        if primary_id:
            worker = self._registry.get_worker(primary_id)
            if worker and worker.is_available and worker.can_serve(request.model):
                return worker

        # Fallback: any available worker that can serve this model
        available = [
            w for w in self._registry.get_available_workers()
            if w.can_serve(request.model)
        ]
        if not available:
            return None

        # Pick least loaded among available
        return min(available, key=lambda w: w.active_requests)

    async def generate(
        self, request: HelixRequest
    ) -> AsyncIterator[str]:
        """
        Route request to a worker and stream tokens back.
        Acquires/releases the worker slot around the generation.
        Raises RuntimeError if no worker is available.
        """
        worker = self.resolve_worker(request)
        if worker is None:
            raise RuntimeError(
                f"No available worker for model={request.model!r}. "
                f"Pool has {len(self._backends)} backends registered."
            )

        backend = self._backends.get(worker.worker_id)
        if backend is None:
            raise RuntimeError(
                f"Backend instance missing for worker {worker.worker_id}"
            )

        worker.acquire()
        request.mark_dispatched(worker.worker_id)
        logger.info(
            "Dispatching request %s to worker %s (active=%d)",
            request.request_id,
            worker.worker_id,
            worker.active_requests,
        )

        try:
            async for token in backend.generate(request):
                request.tokens_generated += 1
                yield token
            request.mark_completed()
        except Exception as exc:
            request.mark_failed()
            logger.error(
                "Worker %s failed on request %s: %s",
                worker.worker_id,
                request.request_id,
                exc,
            )
            raise
        finally:
            worker.release()

    # ── Stats ─────────────────────────────────────────────────

    def pool_stats(self) -> dict[str, object]:
        return {
            "total_backends": len(self._backends),
            "registry": self._registry.status_summary(),
            "ring": self._ring.rebalance_stats(),
        }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _infer_backend_type(url: str) -> BackendType:
        """
        Infer backend type from URL.
        Convention:
          port 11434        → Ollama (default Ollama port)
          port 8001 or mlx  → MLX-Serve
          port 8002 or vllm → vLLM
          default           → Ollama
        """
        url_lower = url.lower()
        if "mlx" in url_lower or ":8001" in url_lower:
            return BackendType.MLX
        if "vllm" in url_lower or ":8002" in url_lower:
            return BackendType.VLLM
        return BackendType.OLLAMA
