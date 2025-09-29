from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRequest:
    question: str
    top_k: int = 5
    use_reranker: bool = True
