"""CLI for admin tasks and bulk ingestion.

Why: Operative Tasks wie Shards/Replicas/Quantization gehören in Admin-CLI,
     nicht in Use-Cases.
"""

import argparse
import sys

from bu_superagent.config.composition import build_embedding, build_vector_store
from bu_superagent.config.settings import AppSettings


def cmd_ensure_collection(args) -> int:
    """Create or ensure collection exists with scaling config.

    Args:
        args: Parsed arguments (collection, dim, shards, replicas, metric)

    Returns:
        Exit code (0=success, 1=failure)
    """
    try:
        settings = AppSettings()
        vs = build_vector_store(settings)

        # Check if adapter supports admin port
        if not hasattr(vs, "ensure_collection"):
            print("Error: Vector store does not support admin operations")
            return 1

        result = vs.ensure_collection(
            name=args.collection,
            dim=args.dim,
            shards=args.shards,
            replicas=args.replicas,
            metric=args.metric,
        )

        if result.ok:
            print(
                f"✓ Collection '{args.collection}' ready "
                f"(dim={args.dim}, shards={args.shards}, replicas={args.replicas})"
            )
            return 0
        else:
            print(f"✗ Failed: {result.error}")
            return 1

    except Exception as ex:
        print(f"✗ Error: {ex}")
        return 1


def cmd_set_quantization(args) -> int:
    """Configure quantization for collection.

    Args:
        args: Parsed arguments (collection, kind, params)

    Returns:
        Exit code (0=success, 1=failure)
    """
    try:
        settings = AppSettings()
        vs = build_vector_store(settings)

        if not hasattr(vs, "set_quantization"):
            print("Error: Vector store does not support quantization")
            return 1

        params = {
            "quantile": 0.99,
            "always_ram": True,
        }

        result = vs.set_quantization(
            name=args.collection,
            kind=args.kind,
            params=params,
        )

        if result.ok:
            print(f"✓ Quantization '{args.kind}' enabled for '{args.collection}'")
            return 0
        else:
            print(f"✗ Failed: {result.error}")
            return 1

    except Exception as ex:
        print(f"✗ Error: {ex}")
        return 1


def cmd_ingest_batch(args) -> int:
    """Ingest documents from file in batches.

    Args:
        args: Parsed arguments (collection, file, batch_size)

    Returns:
        Exit code (0=success, 1=failure)
    """
    try:
        import json

        settings = AppSettings()
        embed = build_embedding(settings)
        vs = build_vector_store(settings)

        # Load documents from file
        with open(args.file, encoding="utf-8") as f:
            docs = json.load(f)

        print(f"Loaded {len(docs)} documents from {args.file}")

        # Process in batches
        batch_size = args.batch_size
        total_chunks = 0

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d["text"] for d in batch]
            ids = [d["id"] for d in batch]
            metas = [d.get("meta", {}) for d in batch]

            # Embed
            result = embed.embed_texts(texts)
            if not result.ok:
                print(f"✗ Embedding failed: {result.error}")
                return 1

            # Upsert
            result = vs.upsert(args.collection, ids, result.value, metas)
            if not result.ok:
                print(f"✗ Upsert failed: {result.error}")
                return 1

            total_chunks += len(batch)
            print(f"  Processed {total_chunks}/{len(docs)} documents")

        print(f"✓ Ingested {total_chunks} documents into '{args.collection}'")
        return 0

    except Exception as ex:
        print(f"✗ Error: {ex}")
        return 1


def cmd_benchmark(args) -> int:
    """Benchmark query performance.

    Args:
        args: Parsed arguments (collection, queries, top_k, iterations)

    Returns:
        Exit code (0=success, 1=failure)
    """
    try:
        import statistics
        import time

        settings = AppSettings()
        embed = build_embedding(settings)
        vs = build_vector_store(settings)

        # Test queries
        queries = args.queries or [
            "What is RAG?",
            "How does vector search work?",
            "Explain embeddings",
        ]

        latencies = []

        print(f"Running {args.iterations} iterations...")

        for _iteration in range(args.iterations):
            for query in queries:
                # Embed query
                start = time.perf_counter()
                result = embed.embed_texts([query])
                if not result.ok:
                    continue

                # Search
                result = vs.search(args.collection, result.value[0], top_k=args.top_k)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if result.ok:
                    latencies.append(elapsed_ms)

        if latencies:
            print("\n✓ Benchmark results:")
            print(f"  Total queries: {len(latencies)}")
            print(f"  Mean latency: {statistics.mean(latencies):.2f} ms")
            print(f"  Median latency: {statistics.median(latencies):.2f} ms")
            print(f"  P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
            print(f"  P99 latency: {statistics.quantiles(latencies, n=100)[98]:.2f} ms")
            return 0
        else:
            print("✗ No successful queries")
            return 1

    except Exception as ex:
        print(f"✗ Error: {ex}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point with subcommands.

    Subcommands:
    - ensure-collection: Create/configure collection with shards/replicas
    - set-quantization: Enable compression (scalar/product/binary)
    - ingest-batch: Bulk ingest from JSON file
    - benchmark: Performance testing

    Returns:
        Exit code (0=success, 1=failure)
    """
    parser = argparse.ArgumentParser(
        prog="bu-superagent-admin",
        description="Admin CLI for BU Superagent RAG system",
    )

    subparsers = parser.add_subparsers(dest="command", help="Admin commands")

    # ensure-collection
    p_ensure = subparsers.add_parser("ensure-collection", help="Create or ensure collection exists")
    p_ensure.add_argument("--collection", required=True, help="Collection name")
    p_ensure.add_argument("--dim", type=int, required=True, help="Vector dimension")
    p_ensure.add_argument("--shards", type=int, default=1, help="Number of shards (default: 1)")
    p_ensure.add_argument("--replicas", type=int, default=1, help="Replication factor (default: 1)")
    p_ensure.add_argument("--metric", default="cosine", help="Distance metric (default: cosine)")

    # set-quantization
    p_quant = subparsers.add_parser("set-quantization", help="Configure quantization")
    p_quant.add_argument("--collection", required=True, help="Collection name")
    p_quant.add_argument(
        "--kind",
        choices=["scalar", "product", "binary"],
        required=True,
        help="Quantization type",
    )

    # ingest-batch
    p_ingest = subparsers.add_parser("ingest-batch", help="Bulk ingest from file")
    p_ingest.add_argument("--collection", required=True, help="Collection name")
    p_ingest.add_argument("--file", required=True, help="JSON file with documents")
    p_ingest.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark query performance")
    p_bench.add_argument("--collection", required=True, help="Collection name")
    p_bench.add_argument("--top-k", type=int, default=5, help="Top K results")
    p_bench.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    p_bench.add_argument("--queries", nargs="+", help="Custom queries")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handlers
    if args.command == "ensure-collection":
        return cmd_ensure_collection(args)
    elif args.command == "set-quantization":
        return cmd_set_quantization(args)
    elif args.command == "ingest-batch":
        return cmd_ingest_batch(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
