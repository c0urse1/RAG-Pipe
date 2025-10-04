import argparse
import os

from bu_superagent.application.dto.ingest_dto import IngestDocumentRequest
from bu_superagent.config.composition import build_ingest_use_case


def main() -> None:
    ap = argparse.ArgumentParser("ingest")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument(
        "--collection",
        default=os.getenv("VECTOR_COLLECTION", "kb_chunks_de_1024d"),
    )
    ap.add_argument("--embedding-kind", default="e5", choices=["e5"])
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--overhang", type=int, default=200)
    ap.add_argument("--merge", type=int, default=500)
    args = ap.parse_args()

    uc = build_ingest_use_case()
    req = IngestDocumentRequest(
        doc_id=args.doc_id,
        path=args.path,
        collection=args.collection,
        embedding_kind=args.embedding_kind,
        target_chars=args.target,
        overlap_chars=args.overlap,
        max_overhang=args.overhang,
        merge_threshold=args.merge,
        inject_section_titles=True,
    )
    n = uc.execute(req)
    print(f"Ingest done: {n} chunks")


if __name__ == "__main__":
    main()
