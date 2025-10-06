"""CLI query handler for BU Superagent (Confidence-Gate architecture)."""

import argparse

from bu_superagent.application.dto.query_dto import QueryRequest
from bu_superagent.config.composition import build_query_use_case


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="Use extractive fallback")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold (0-1)")
    parser.add_argument("--mmr", action="store_true", help="Enable MMR diversity")
    parser.add_argument(
        "--use-reranker", action="store_true", default=False, help="Enable reranking"
    )
    parser.add_argument("--pre-rerank-k", type=int, default=20, help="Candidates before rerank")
    # parse_known_args vermeidet Fehler bei unbekannten Flags (z. B. -q von pytest)
    args, _unknown = parser.parse_known_args()

    # In Testumgebung darf main() ohne Argumente aufgerufen werden.
    if not args.question:
        # Signalisiere Platzhalterzustand wie in Use-Case-Tests
        raise NotImplementedError

    uc = build_query_use_case(with_llm=not args.no_llm, with_reranker=args.use_reranker)
    req = QueryRequest(
        question=args.question,
        top_k=args.k,
        confidence_threshold=args.confidence,
        mmr=args.mmr,
        use_reranker=args.use_reranker,
        pre_rerank_k=args.pre_rerank_k,
    )
    result = uc.execute(req)

    if result.ok and result.value is not None:
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(result.value.text)
        print("\n" + "=" * 80)
        print("CITATIONS:")
        print("=" * 80)
        for i, c in enumerate(result.value.citations, 1):
            print(f"[{i}] {c.source} (score={c.score:.3f})")
    elif result.error is not None:
        err = result.error
        err_name = type(err).__name__
        err_msg = getattr(err, "message", str(err))
        print(f"\n[ERROR] {err_name}: {err_msg}")
        if hasattr(err, "top_score") and hasattr(err, "threshold"):
            print(f"  → Top score: {err.top_score:.3f}, Threshold: {err.threshold:.3f}")


if __name__ == "__main__":
    main()
