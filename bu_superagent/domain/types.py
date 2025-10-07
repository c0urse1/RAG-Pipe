from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


@dataclass(frozen=True)
class Result(Generic[T, E]):
    ok: bool
    value: T | None = None
    error: E | None = None

    @staticmethod
    def success(v: T) -> "Result[T, E]":
        return Result(ok=True, value=v)

    @staticmethod
    def failure(e: E) -> "Result[T, E]":
        return Result(ok=False, error=e)


Vector = tuple[float, ...]  # 1024-d for e5; domain stays agnostic to dim but validate elsewhere
Score = float


@dataclass(frozen=True)
class DocumentId:
    value: str


@dataclass(frozen=True)
class ChunkId:
    value: str


@dataclass(frozen=True)
class ShardKey:
    value: str  # e.g., tenant/project/date-bucket


@dataclass(frozen=True)
class Chunk:
    id: ChunkId
    doc_id: DocumentId
    text: str
    vector: Vector | None
    meta: dict[str, Any]
