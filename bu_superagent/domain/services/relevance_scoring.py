from __future__ import annotations

import math
from typing import Iterable


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    va = list(a)
    vb = list(b)
    if len(va) != len(vb) or len(va) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0 or nb == 0:
        return 0.0
    # map from [-1,1] to [0,1]
    return (dot / (na * nb) + 1.0) / 2.0
