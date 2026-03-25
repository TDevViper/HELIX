from __future__ import annotations
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HELIX_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Workers
    bootstrap_workers: List[str] = ["http://localhost:11434"]
    worker_request_timeout_seconds: int = 120

    # Scheduler
    scheduler_poll_interval_ms: int = 50
    preemption_enabled: bool = True

    # Memory pressure thresholds (fraction of total memory)
    memory_elevated_threshold: float = 0.70
    memory_high_threshold: float = 0.80
    memory_critical_threshold: float = 0.90

    # Gossip
    gossip_interval_ms: int = 500
    gossip_fan_out: int = 3
    gossip_suspect_threshold: int = 3

    # Auth
    api_keys: List[str] = []
    auth_enabled: bool = False

    # Rate limiting (requests per minute per client)
    rate_limit_rpm: int = 60
    rate_limit_enabled: bool = True

    # Observability
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    metrics_enabled: bool = True


settings = Settings()
