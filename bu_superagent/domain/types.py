from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


@dataclass(frozen=True)
class Result(Generic[T, E]):
    value: Optional[T] = None
    error: Optional[E] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None
