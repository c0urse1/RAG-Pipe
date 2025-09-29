from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Score:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Score must be between 0 and 1")
