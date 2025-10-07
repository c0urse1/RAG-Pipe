"""Telemetry port for monitoring and metrics."""

from typing import Any, Protocol


class TelemetryPort(Protocol):
    """Port for telemetry and monitoring."""

    def incr(self, name: str, tags: dict[str, Any] | None = None) -> None:
        """Increment a counter metric."""
        ...

    def observe(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
        """Observe a value for histogram/summary metric."""
        ...
