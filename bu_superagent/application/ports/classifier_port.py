from dataclasses import dataclass
from typing import Protocol
from collections.abc import Sequence


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


class ClassifierPort(Protocol):
    def classify(
        self, texts: Sequence[str], candidate_labels: Sequence[str]
    ) -> list[list[LabelScore]]: ...
