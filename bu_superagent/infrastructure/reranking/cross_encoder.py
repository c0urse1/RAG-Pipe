from __future__ import annotations

from collections.abc import Sequence

from ...application.ports.reranker_port import RerankerPort


class SimpleOverlapReranker(RerankerPort):
    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        scores: list[float] = []
        for c in candidates:
            c_tokens = set(c.lower().split())
            if not c_tokens:
                scores.append(0.0)
                continue
            inter = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            jaccard = inter / union if union else 0.0
            scores.append(jaccard)
        return scores
