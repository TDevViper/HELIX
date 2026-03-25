from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HELIX_HOST")
    port: int = Field(default=8000, alias="HELIX_PORT")
    debug: bool = Field(default=False, alias="HELIX_DEBUG")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="HELIX_LOG_LEVEL"
    )

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0", alias="HELIX_REDIS_URL"
    )
    redis_prefix: str = Field(default="helix:", alias="HELIX_REDIS_PREFIX")

    # ── Scheduler ─────────────────────────────────────────────
    scheduler_tick_ms: int = Field(
        default=50, alias="HELIX_SCHEDULER_TICK_MS"
    )
    max_queue_depth: int = Field(
        default=512, alias="HELIX_MAX_QUEUE_DEPTH"
    )
    preemption_enabled: bool = Field(
        default=True, alias="HELIX_PREEMPTION_ENABLED"
    )
    checkpoint_ttl_seconds: int = Field(
        default=300, alias="HELIX_CHECKPOINT_TTL_SECONDS"
    )

    # ── Hash Ring ─────────────────────────────────────────────
    virtual_nodes: int = Field(
        default=150, alias="HELIX_VIRTUAL_NODES"
    )

    # ── Gossip ────────────────────────────────────────────────
    gossip_interval_ms: int = Field(
        default=500, alias="HELIX_GOSSIP_INTERVAL_MS"
    )
    gossip_suspect_threshold: int = Field(
        default=3, alias="HELIX_GOSSIP_SUSPECT_THRESHOLD"
    )
    gossip_fan_out: int = Field(
        default=3, alias="HELIX_GOSSIP_FAN_OUT"
    )

    # ── Memory Pressure ───────────────────────────────────────
    memory_poll_interval_seconds: int = Field(
        default=2, alias="HELIX_MEMORY_POLL_INTERVAL_SECONDS"
    )
    memory_elevated_threshold: float = Field(
        default=0.65, alias="HELIX_MEMORY_ELEVATED_THRESHOLD"
    )
    memory_high_threshold: float = Field(
        default=0.80, alias="HELIX_MEMORY_HIGH_THRESHOLD"
    )
    memory_critical_threshold: float = Field(
        default=0.92, alias="HELIX_MEMORY_CRITICAL_THRESHOLD"
    )

    # ── KV Cache ──────────────────────────────────────────────
    prefix_cache_enabled: bool = Field(
        default=True, alias="HELIX_PREFIX_CACHE_ENABLED"
    )
    prefix_cache_ttl_seconds: int = Field(
        default=3600, alias="HELIX_PREFIX_CACHE_TTL_SECONDS"
    )
    max_cached_prefixes: int = Field(
        default=10_000, alias="HELIX_MAX_CACHED_PREFIXES"
    )

    # ── Auth ──────────────────────────────────────────────────
    api_keys: list[str] = Field(
        default=["dev-key-local"], alias="HELIX_API_KEYS"
    )
    auth_enabled: bool = Field(
        default=False, alias="HELIX_AUTH_ENABLED"
    )

    # ── Rate Limiting ─────────────────────────────────────────
    rate_limit_enabled: bool = Field(
        default=True, alias="HELIX_RATE_LIMIT_ENABLED"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60, alias="HELIX_RATE_LIMIT_RPM"
    )

    # ── Observability ─────────────────────────────────────────
    otel_enabled: bool = Field(
        default=False, alias="HELIX_OTEL_ENABLED"
    )
    otel_endpoint: str = Field(
        default="http://localhost:4317", alias="HELIX_OTEL_ENDPOINT"
    )
    metrics_enabled: bool = Field(
        default=True, alias="HELIX_METRICS_ENABLED"
    )

    # ── Workers (bootstrap) ───────────────────────────────────
    # Comma-separated list of worker URLs for static bootstrap
    # e.g. "http://localhost:11434,http://localhost:8001"
    bootstrap_workers: list[str] = Field(
        default=[], alias="HELIX_BOOTSTRAP_WORKERS"
    )
    worker_request_timeout_seconds: int = Field(
        default=120, alias="HELIX_WORKER_TIMEOUT_SECONDS"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


# Module-level singleton — import this everywhere
settings = Settings()
