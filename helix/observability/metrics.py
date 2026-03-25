from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Counters ──────────────────────────────────────────────────
requests_total = Counter(
    "helix_requests_total",
    "Total requests received",
    ["model", "tier"],
)
requests_failed = Counter(
    "helix_requests_failed_total",
    "Total failed requests",
    ["model", "reason"],
)
tokens_generated = Counter(
    "helix_tokens_generated_total",
    "Total tokens generated across all workers",
    ["worker_id"],
)
cache_hits = Counter(
    "helix_prefix_cache_hits_total",
    "Prefix cache hits",
)
cache_misses = Counter(
    "helix_prefix_cache_misses_total",
    "Prefix cache misses",
)
preemptions = Counter(
    "helix_preemptions_total",
    "Requests preempted under memory pressure",
)

# ── Gauges ────────────────────────────────────────────────────
queue_depth = Gauge(
    "helix_queue_depth",
    "Current scheduler queue depth",
)
active_requests = Gauge(
    "helix_active_requests",
    "In-flight requests across all workers",
)
healthy_workers = Gauge(
    "helix_healthy_workers",
    "Number of healthy workers in the registry",
)

# ── Histograms ────────────────────────────────────────────────
dispatch_latency = Histogram(
    "helix_dispatch_latency_ms",
    "Time from request arrival to first dispatch (ms)",
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)
request_duration = Histogram(
    "helix_request_duration_ms",
    "End-to-end request duration (ms)",
    buckets=[100, 500, 1000, 2500, 5000, 10000, 30000],
)
