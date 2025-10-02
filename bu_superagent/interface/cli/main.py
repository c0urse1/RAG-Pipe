import argparse

from bu_superagent.application.dto.query_dto import QueryRequest
from bu_superagent.config.composition import build_query_use_case


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--question")
    parser.add_argument("--k", type=int, default=5)
    # parse_known_args vermeidet Fehler bei unbekannten Flags (z. B. -q von pytest)
    args, _unknown = parser.parse_known_args()

    # In Testumgebung darf main() ohne Argumente aufgerufen werden.
    if not args.question:
        # Signalisiere Platzhalterzustand wie in Use-Case-Tests
        raise NotImplementedError

    uc = build_query_use_case()
    req = QueryRequest(question=args.question, top_k=args.k)
    # Achtung: Use-Case.execute ist noch Platzhalter (SAM-konform vorher definiert)
    uc.execute(req)  # später: Ergebnis formatieren/ausgeben


if __name__ == "__main__":
    main()
