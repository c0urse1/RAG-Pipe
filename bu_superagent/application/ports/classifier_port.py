from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


class ClassifierPort(Protocol):
    def classify(
        self, texts: Sequence[str], candidate_labels: Sequence[str]
    ) -> list[list[LabelScore]]: ...
