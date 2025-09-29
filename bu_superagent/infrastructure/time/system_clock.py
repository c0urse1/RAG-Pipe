from __future__ import annotations

from datetime import datetime, timezone

from ...application.ports.clock_port import ClockPort


class SystemClock(ClockPort):
    def now(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)
