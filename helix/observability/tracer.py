from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "helix") -> None:
    """
    Configure OpenTelemetry tracing. No-ops gracefully if otel_enabled=False
    or if the OTel packages are not installed.
    """
    from helix.config import settings

    if not settings.otel_enabled:
        logger.info("OTel tracing disabled (HELIX_OTEL_ENABLED=false)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OTel tracing configured (endpoint=%s)", settings.otel_endpoint)
    except ImportError:
        logger.warning("opentelemetry packages not installed — tracing disabled")
