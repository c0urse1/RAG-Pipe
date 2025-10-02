from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DocumentPayload:
    text: str
    title: str | None = None
    source_path: str | None = None


class DocumentLoaderPort(Protocol):
    def load(self, path: str) -> DocumentPayload: ...
