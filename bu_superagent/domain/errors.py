from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DomainError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ValidationError(DomainError):
    pass
