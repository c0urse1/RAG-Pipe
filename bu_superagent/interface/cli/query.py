"""Standalone CLI query example for BU Superagent.

This module demonstrates the Confidence-Gate architecture pattern:
- Interface layer is thin (parse args, format output)
- All orchestration lives in application/domain
- Use case returns Result[RAGAnswer, Error] for explicit error handling
"""

from bu_superagent.application.dto.query_dto import QueryRequest
from bu_superagent.config.composition import build_query_use_case


def main() -> None:
    """Execute a query against the knowledge base.

    Example usage:
        python -m bu_superagent.interface.cli.query

    The query demonstrates:
    - Building use case via composition root (config layer)
    - Executing with QueryRequest DTO
    - Handling Result type (success or failure)
    - Formatting output for CLI presentation
    """
    # Build use case from composition root (wires all dependencies)
    uc = build_query_use_case(with_llm=True)

    # Create request DTO with query parameters
    req = QueryRequest(
        question="Was ist die Wartezeit-Regelung für Bestandskunden?",
        top_k=5,
        confidence_threshold=0.35,
        mmr=True,  # Enable diversity via Maximal Marginal Relevance
        mmr_lambda=0.5,  # Balance relevance vs. diversity
    )

    # Execute use case (returns Result[RAGAnswer, Exception])
    result = uc.execute(req)

    # Handle result (explicit success/failure pattern)
    if result.ok and result.value is not None:
        # Success path: format and display answer
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(result.value.text)

        print("\n" + "=" * 80)
        print("CITATIONS:")
        print("=" * 80)
        for i, citation in enumerate(result.value.citations, 1):
            print(f"[{i}] {citation.source} (score={citation.score:.3f})")

    elif result.error is not None:
        # Failure path: format error for user
        err = result.error
        err_name = type(err).__name__
        err_msg = getattr(err, "message", str(err))

        print(f"\n[ERROR] {err_name}: {err_msg}")

        # Special handling for LowConfidenceError (domain-specific)
        if hasattr(err, "top_score") and hasattr(err, "threshold"):
            print(f"  → Top score: {err.top_score:.3f}, Threshold: {err.threshold:.3f}")
            print("  → Recommendation: Escalate to human review or refine query")


if __name__ == "__main__":
    main()
