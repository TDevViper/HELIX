from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BackendType(str, Enum):
    OLLAMA = "ollama"
    MLX = "mlx"
    VLLM = "vllm"


class WorkerStatus(str, Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected"   # Gossip: missed pings but not confirmed dead
    DEAD = "dead"             # Confirmed dead, removed from ring
    DRAINING = "draining"     # Graceful shutdown, no new requests


class WorkerStats(BaseModel):
    worker_id: str
    requests_active: int = 0
    requests_total: int = 0
    tokens_per_second: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    queue_depth: int = 0
    uptime_seconds: float = 0.0

    @property
    def memory_utilization(self) -> float:
        if self.memory_total_gb == 0:
            return 0.0
        return self.memory_used_gb / self.memory_total_gb

    @property
    def is_overloaded(self) -> bool:
        return self.memory_utilization > 0.90 or self.queue_depth > 32


class WorkerNode(BaseModel):
    # ── Identity ──────────────────────────────────────────────
    worker_id: str
    base_url: str                        # e.g. http://localhost:11434
    backend_type: BackendType

    # ── Status ────────────────────────────────────────────────
    status: WorkerStatus = WorkerStatus.HEALTHY
    suspect_count: int = 0               # Incremented by gossip misses
    last_seen_at: float = Field(
        default_factory=lambda: time.time()
    )
    registered_at: float = Field(
        default_factory=lambda: time.time()
    )

    # ── Capacity ──────────────────────────────────────────────
    weight: int = 1                      # Hash ring weight (vnodes multiplier)
    max_concurrent_requests: int = 8
    active_requests: int = 0

    # ── Live stats (updated by health poller) ─────────────────
    stats: Optional[WorkerStats] = None

    # ── Models this worker can serve ──────────────────────────
    supported_models: list[str] = Field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.status == WorkerStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """Healthy and has capacity for another request."""
        return (
            self.is_healthy
            and self.active_requests < self.max_concurrent_requests
        )

    @property
    def memory_used_gb(self) -> float:
        return self.stats.memory_used_gb if self.stats else 0.0

    @property
    def memory_utilization(self) -> float:
        return self.stats.memory_utilization if self.stats else 0.0

    def touch(self) -> None:
        """Update last_seen_at on successful ping."""
        self.last_seen_at = time.time()
        self.suspect_count = 0
        self.status = WorkerStatus.HEALTHY

    def increment_suspect(self, threshold: int = 3) -> bool:
        """
        Called by gossip on missed ping.
        Returns True if worker just crossed into DEAD status.
        """
        self.suspect_count += 1
        if self.suspect_count >= threshold:
            if self.status != WorkerStatus.DEAD:
                self.status = WorkerStatus.DEAD
                return True
        elif self.suspect_count >= 1:
            self.status = WorkerStatus.SUSPECTED
        return False

    def can_serve(self, model: str) -> bool:
        """True if this worker can handle requests for the given model."""
        if not self.supported_models:
            return True   # Empty list = serves all models
        return model in self.supported_models

    def acquire(self) -> None:
        """Mark one request slot as taken."""
        self.active_requests += 1

    def release(self) -> None:
        """Release one request slot."""
        self.active_requests = max(0, self.active_requests - 1)

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "base_url": self.base_url,
            "backend_type": self.backend_type.value,
            "status": self.status.value,
            "active_requests": self.active_requests,
            "memory_utilization": round(self.memory_utilization, 3),
            "supported_models": self.supported_models,
        }
