from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class IngestDocumentDTO:
    id: str
    title: str
    content: str
