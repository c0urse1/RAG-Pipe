from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime


class ClockPort(ABC):
    @abstractmethod
    def now(self) -> datetime: ...


class SystemUTCClock(ClockPort):
    def now(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(UTC)
