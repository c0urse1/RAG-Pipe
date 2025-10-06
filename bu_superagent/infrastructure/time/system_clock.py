"""System clock adapter providing real UTC time.

This is the production implementation of ClockPort.
For tests, inject FakeClock or similar test doubles.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ...application.ports.clock_port import ClockPort


class SystemClock(ClockPort):
    """Production clock adapter returning real system time in UTC.

    Why (SAM): Infrastructure adapters implement ports. This is the only
    concrete clock implementation. Tests should use fakes/stubs, not this.
    """

    def now(self) -> datetime:  # pragma: no cover - trivial
        """Return current system time in UTC timezone."""
        return datetime.now(UTC)
