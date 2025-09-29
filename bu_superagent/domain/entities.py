from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import DocumentId, ChunkId


@dataclass(slots=True, frozen=True)
class Document:
    id: DocumentId
    title: str
    content: str


@dataclass(slots=True, frozen=True)
class Chunk:
    id: ChunkId
    document_id: DocumentId
    text: str
    embedding: List[float] | None = None
