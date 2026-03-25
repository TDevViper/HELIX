from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerStats


class LLMBackend(ABC):
    """
    Abstract base class for all LLM backend adapters.
    Every backend (Ollama, MLX-Serve, vLLM) must implement this interface.
    The scheduler and worker pool only talk to this ABC — never to
    concrete backends directly.
    """

    # ── Identity ──────────────────────────────────────────────

    @property
    @abstractmethod
    def worker_id(self) -> str:
        """Unique identifier for this worker instance."""
        ...

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Which backend technology this adapter wraps."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL of the backend server."""
        ...

    # ── Core inference ────────────────────────────────────────

    @abstractmethod
    async def generate(
        self, request: HelixRequest
    ) -> AsyncIterator[str]:
        """
        Stream tokens for the given request.
        Yields raw token strings one at a time.
        Must be an async generator — yields until completion or cancellation.
        Raises RuntimeError on unrecoverable backend error.
        """
        ...

    # ── Health & stats ────────────────────────────────────────

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Returns True if backend is reachable and ready to serve.
        Must complete within 2 seconds — used by gossip loop.
        Must never raise — return False on any exception.
        """
        ...

    @abstractmethod
    async def get_stats(self) -> WorkerStats:
        """
        Return current resource stats for this worker.
        Called by MemoryMonitor every poll interval.
        Must never raise — return zeroed WorkerStats on failure.
        """
        ...

    @property
    @abstractmethod
    def memory_used_gb(self) -> float:
        """Last known memory usage in GB. Cached from get_stats()."""
        ...

    # ── Lifecycle ─────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Optional: called once when backend is registered.
        Use for connection pool warmup, model preload checks, etc.
        Default is no-op.
        """

    async def shutdown(self) -> None:
        """
        Optional: called on graceful shutdown or worker drain.
        Use to close HTTP sessions, flush buffers, etc.
        Default is no-op.
        """

    # ── Helpers (concrete, shared by all backends) ────────────

    def supports_model(self, model: str, supported: list[str]) -> bool:
        """Check if this backend can serve the requested model."""
        if not supported:
            return True  # Empty list = serve all
        return model in supported

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"worker_id={self.worker_id!r}, "
            f"backend={self.backend_type.value}, "
            f"url={self.base_url!r})"
        )
