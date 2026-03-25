from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class Priority(int, Enum):
    """Higher value = higher urgency. Used for heap ordering."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class UserTier(str, Enum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"


class SchedulerState(str, Enum):
    PENDING = "pending"        # In queue, not yet dispatched
    DISPATCHED = "dispatched"  # Sent to a worker
    STREAMING = "streaming"    # Worker is generating tokens
    PREEMPTED = "preempted"    # Pulled back under memory pressure
    RESUMED = "resumed"        # Requeued after preemption
    COMPLETED = "completed"    # Done successfully
    FAILED = "failed"          # Terminal failure


class Message(BaseModel):
    role: str
    content: str


class HelixRequest(BaseModel):
    # ── Identity ──────────────────────────────────────────────
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = "anonymous"

    # ── Payload ───────────────────────────────────────────────
    model: str = "llama3"
    messages: list[Message] = Field(default_factory=list)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)

    # ── Scheduling ────────────────────────────────────────────
    priority: Priority = Priority.NORMAL
    user_tier: UserTier = UserTier.STANDARD
    deadline_ms: float = Field(
        default_factory=lambda: (time.time() + 30) * 1000  # 30s default
    )
    arrived_at: float = Field(default_factory=lambda: time.time() * 1000)

    # ── State ─────────────────────────────────────────────────
    state: SchedulerState = SchedulerState.PENDING
    assigned_worker_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # ── Preemption / Resume ───────────────────────────────────
    is_resumed: bool = False
    checkpoint_key: Optional[str] = None  # Redis key for KV checkpoint

    # ── Cache ─────────────────────────────────────────────────
    prefix_hash: Optional[str] = None
    prefix_cache_hit: bool = False
    prefill_skipped: bool = False

    # ── Observability ─────────────────────────────────────────
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tokens_generated: int = 0
    dispatch_latency_ms: float = 0.0  # time from arrived_at to dispatch

    def priority_score(self) -> float:
        """
        Compute scheduling priority score. Higher = more urgent.
        Used by the preemptive scheduler heap.
        """
        tier_bonus = {
            UserTier.PREMIUM: 2.0,
            UserTier.STANDARD: 1.0,
            UserTier.FREE: 0.5,
        }
        now_ms = time.time() * 1000
        deadline_urgency = 1.0 / max(self.deadline_ms - now_ms, 1.0)
        size_penalty = self.max_tokens / 4096.0
        return tier_bonus[self.user_tier] * deadline_urgency / size_penalty

    def is_expired(self) -> bool:
        """True if past deadline — should be dropped, not dispatched."""
        return time.time() * 1000 > self.deadline_ms

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def mark_dispatched(self, worker_id: str) -> None:
        self.state = SchedulerState.DISPATCHED
        self.assigned_worker_id = worker_id
        self.dispatch_latency_ms = (time.time() * 1000) - self.arrived_at

    def mark_preempted(self, checkpoint_key: str) -> None:
        self.state = SchedulerState.PREEMPTED
        self.checkpoint_key = checkpoint_key
        self.is_resumed = True
        self.retry_count += 1

    def mark_completed(self) -> None:
        self.state = SchedulerState.COMPLETED

    def mark_failed(self) -> None:
        self.state = SchedulerState.FAILED

    def __lt__(self, other: "HelixRequest") -> bool:
        """Heap ordering: higher priority_score = smaller heap value."""
        return self.priority_score() > other.priority_score()

    class Config:
        use_enum_values = False
