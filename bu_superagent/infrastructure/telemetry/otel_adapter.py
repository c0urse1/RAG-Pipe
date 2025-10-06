"""OpenTelemetry adapter for metrics and tracing.

Why: Messbarkeit (throughput, p95 latency, recall@k) ist Pflicht für Scale.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from bu_superagent.application.ports import TelemetryPort


@dataclass
class OtelConfig:
    """Configuration for OpenTelemetry."""

    service_name: str = "bu-superagent"
    otlp_endpoint: str | None = None  # e.g., "http://localhost:4317"
    environment: str = "production"
    enable_console: bool = False  # Debug: print metrics to console


class OpenTelemetryAdapter(TelemetryPort):
    """OpenTelemetry adapter for metrics and distributed tracing.

    Metrics:
    - Counters: incr() for events (queries, errors, cache hits)
    - Histograms: observe() for distributions (latency, chunk counts)

    Tags/Attributes:
    - Environment, service, endpoint, status, error_type
    - Enables filtering and aggregation in observability tools

    Why: Messbarkeit (throughput, p95 latency, recall@k) ist Pflicht für Scale.
         OpenTelemetry provides vendor-neutral observability.
    """

    def __init__(self, cfg: OtelConfig) -> None:
        """Initialize OpenTelemetry adapter.

        Args:
            cfg: OtelConfig with service name and OTLP endpoint

        Note: Gracefully handles missing opentelemetry-sdk dependency.
              Metrics become no-ops if library not installed.
        """
        self._cfg = cfg
        self._meter: Any | None = None
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry SDK with lazy import.

        Sets up:
        - OTLP exporter (if endpoint configured)
        - Console exporter (if enable_console=True)
        - Meter for creating instruments
        """
        try:
            # Lazy imports
            otel_sdk = import_module("opentelemetry.sdk.metrics")
            otel_metrics = import_module("opentelemetry.metrics")
            otel_resources = import_module("opentelemetry.sdk.resources")

            # Build resource attributes
            Resource = otel_resources.Resource
            resource = Resource.create(
                {
                    "service.name": self._cfg.service_name,
                    "deployment.environment": self._cfg.environment,
                }
            )

            # Build metric readers
            readers = []

            # OTLP exporter (production)
            if self._cfg.otlp_endpoint:
                otel_otlp = import_module("opentelemetry.exporter.otlp.proto.grpc.metric_exporter")
                otlp_exporter = otel_otlp.OTLPMetricExporter(endpoint=self._cfg.otlp_endpoint)
                PeriodicExportingMetricReader = otel_sdk.export.PeriodicExportingMetricReader
                readers.append(PeriodicExportingMetricReader(otlp_exporter))

            # Console exporter (debug)
            if self._cfg.enable_console:
                otel_console = import_module("opentelemetry.sdk.metrics.export")
                console_exporter = otel_console.ConsoleMetricExporter()
                PeriodicExportingMetricReader = otel_sdk.export.PeriodicExportingMetricReader
                readers.append(PeriodicExportingMetricReader(console_exporter))

            # Create meter provider
            MeterProvider = otel_sdk.MeterProvider
            provider = MeterProvider(resource=resource, metric_readers=readers)
            otel_metrics.set_meter_provider(provider)

            # Get meter for creating instruments
            self._meter = otel_metrics.get_meter(__name__)

        except Exception:
            # Gracefully degrade: metrics become no-ops
            # Allows running without opentelemetry-sdk installed
            self._meter = None

    def incr(self, name: str, tags: dict) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name (e.g., "rag.queries.total")
            tags: Attributes for filtering (e.g., {"status": "success"})

        Examples:
            - incr("rag.queries.total", {"status": "success"})
            - incr("rag.cache.hits", {"layer": "embedding"})
            - incr("rag.errors.total", {"error_type": "low_confidence"})
        """
        if self._meter is None:
            return  # No-op if OTel not initialized

        try:
            # Lazy create counter
            if name not in self._counters:
                self._counters[name] = self._meter.create_counter(
                    name=name,
                    description=f"Counter for {name}",
                )

            # Increment with attributes
            self._counters[name].add(1, attributes=tags)

        except Exception:
            # Gracefully handle metric errors (never crash business logic)
            pass

    def observe(self, name: str, value: float, tags: dict) -> None:
        """Observe a value for histogram/summary metric.

        Args:
            name: Metric name (e.g., "rag.query.latency_ms")
            value: Observed value (e.g., 123.45)
            tags: Attributes for filtering (e.g., {"endpoint": "/v1/query"})

        Examples:
            - observe("rag.query.latency_ms", 123.45, {"endpoint": "/v1/query"})
            - observe("rag.chunks.retrieved", 5, {"use_mmr": "true"})
            - observe("rag.confidence.score", 0.85, {"passed_gate": "true"})
        """
        if self._meter is None:
            return  # No-op if OTel not initialized

        try:
            # Lazy create histogram
            if name not in self._histograms:
                self._histograms[name] = self._meter.create_histogram(
                    name=name,
                    description=f"Histogram for {name}",
                )

            # Record value with attributes
            self._histograms[name].record(value, attributes=tags)

        except Exception:
            # Gracefully handle metric errors (never crash business logic)
            pass
