from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class DocumentPayload:
    text: str
    title: Optional[str] = None
    source_path: Optional[str] = None


class DocumentLoaderPort(Protocol):
    def load(self, path: str) -> DocumentPayload: ...
