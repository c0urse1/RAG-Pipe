from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class QueryDTO:
    text: str
    top_k: int = 5
