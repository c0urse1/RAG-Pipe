from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Union, NewType

T = TypeVar("T")
E = TypeVar("E")


DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)


@dataclass(slots=True)
class Ok(Generic[T]):
    value: T


@dataclass(slots=True)
class Err(Generic[E]):
    error: E


Result = Union[Ok[T], Err[E]]
