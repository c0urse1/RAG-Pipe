from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class ClockPort(ABC):
    """Port for time-related operations.

    Why (SAM): Domain/application logic needs testable time operations.
    Infrastructure provides concrete implementations (SystemClock).
    """

    @abstractmethod
    def now(self) -> datetime:
        """Return current UTC datetime."""
        ...
