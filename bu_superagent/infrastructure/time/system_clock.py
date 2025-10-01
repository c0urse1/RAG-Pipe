from __future__ import annotations

from datetime import UTC, datetime

from ...application.ports.clock_port import ClockPort


class SystemClock(ClockPort):
    def now(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(UTC)
