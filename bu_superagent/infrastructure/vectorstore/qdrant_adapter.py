"""Qdrant vector store adapter with scaling capabilities.

Why: Qdrant-Cluster bringt Sharding/Replication/Quantization;
     Adapter kapselt alle externen Typen und wirft nur Domain-Fehler.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from bu_superagent.application.ports import VectorStoreAdminPort, VectorStorePort
from bu_superagent.domain.errors import DomainError, VectorStoreError
from bu_superagent.domain.types import Result, Vector


@dataclass
class QdrantConfig:
    """Configuration for Qdrant client connection."""

    url: str
    api_key: str | None = None
    prefer_grpc: bool = False
    timeout_s: int = 30


class QdrantVectorStoreAdapter(VectorStorePort, VectorStoreAdminPort):
    """Qdrant adapter with admin capabilities for scaling.

    Implements both VectorStorePort (CRUD) and VectorStoreAdminPort
    (sharding, replication, quantization) for production-ready scaling.

    Why: Encapsulates qdrant-client library, converts exceptions to
         domain errors, provides sharding/replication/quantization controls.
    """

    def __init__(self, cfg: QdrantConfig) -> None:
        """Initialize Qdrant adapter with configuration.

        Args:
            cfg: QdrantConfig with connection parameters

        Raises:
            VectorStoreError: If qdrant-client initialization fails
        """
        self._cfg = cfg
        self._client = self._init_client(cfg)

    def _init_client(self, cfg: QdrantConfig) -> Any:
        """Initialize Qdrant client with lazy import.

        Args:
            cfg: QdrantConfig with connection parameters

        Returns:
            QdrantClient instance

        Raises:
            VectorStoreError: If qdrant-client not available or init fails
        """
        try:
            qdrant_client = import_module("qdrant_client")
            return qdrant_client.QdrantClient(
                url=cfg.url,
                api_key=cfg.api_key,
                timeout=cfg.timeout_s,
                prefer_grpc=cfg.prefer_grpc,
            )
        except Exception as ex:
            raise VectorStoreError(f"Qdrant init failed: {ex}") from ex

    def ensure_collection(
        self,
        name: str,
        dim: int,
        shards: int,
        replicas: int,
        metric: str = "cosine",
    ) -> Result[None, DomainError]:
        """Create or ensure collection exists with scaling configuration.

        Args:
            name: Collection name
            dim: Vector dimension
            shards: Number of shards for horizontal scaling
            replicas: Replication factor for high availability
            metric: Distance metric ("cosine", "euclid", "dot")

        Returns:
            Result with None on success or VectorStoreError on failure
        """
        try:
            models = import_module("qdrant_client.models")

            # Map metric string to Qdrant Distance enum
            metric_map = {
                "cosine": models.Distance.COSINE,
                "euclid": models.Distance.EUCLID,
                "dot": models.Distance.DOT,
            }
            distance = metric_map.get(metric.lower(), models.Distance.COSINE)

            # Check if collection exists
            collections = self._client.get_collections()
            exists = any(c.name == name for c in collections.collections)

            if exists:
                # Collection exists, verify configuration
                info = self._client.get_collection(name)
                if info.config.params.vectors.size != dim:
                    return Result.failure(
                        VectorStoreError(
                            f"Collection '{name}' exists with wrong dimension: "
                            f"{info.config.params.vectors.size} != {dim}"
                        )
                    )
            else:
                # Create collection with sharding/replication
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=distance,
                    ),
                    shard_number=shards,
                    replication_factor=replicas,
                )

            return Result.success(None)

        except Exception as ex:
            return Result.failure(VectorStoreError(f"ensure_collection: {ex}"))

    def set_quantization(self, name: str, kind: str, params: dict) -> Result[None, DomainError]:
        """Set quantization configuration for collection.

        Args:
            name: Collection name
            kind: Quantization type ("scalar", "product", "binary")
            params: Quantization parameters (depends on kind)

        Returns:
            Result with None on success or VectorStoreError on failure
        """
        try:
            models = import_module("qdrant_client.models")

            # Map quantization kind to Qdrant config
            if kind == "scalar":
                quant_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=params.get("quantile", 0.99),
                        always_ram=params.get("always_ram", True),
                    )
                )
            elif kind == "product":
                quant_config = models.ProductQuantization(
                    product=models.ProductQuantizationConfig(
                        compression=models.CompressionRatio.X16,
                        always_ram=params.get("always_ram", True),
                    )
                )
            elif kind == "binary":
                quant_config = models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=params.get("always_ram", True),
                    )
                )
            else:
                return Result.failure(VectorStoreError(f"Unknown quantization kind: {kind}"))

            # Update collection quantization
            self._client.update_collection(
                collection_name=name,
                quantization_config=quant_config,
            )

            return Result.success(None)

        except Exception as ex:
            return Result.failure(VectorStoreError(f"set_quantization: {ex}"))

    def set_search_params(self, name: str, params: dict) -> Result[None, DomainError]:
        """Set search parameters for collection.

        Args:
            name: Collection name
            params: Search params (e.g., {"hnsw_ef": 128, "exact": False})

        Returns:
            Result with None on success or VectorStoreError on failure
        """
        try:
            models = import_module("qdrant_client.models")

            # Build HNSW config from params
            hnsw_config = models.HnswConfigDiff(
                m=params.get("hnsw_m"),
                ef_construct=params.get("hnsw_ef_construct"),
                full_scan_threshold=params.get("full_scan_threshold"),
            )

            # Update collection optimizer config
            self._client.update_collection(
                collection_name=name,
                hnsw_config=hnsw_config,
            )

            return Result.success(None)

        except Exception as ex:
            return Result.failure(VectorStoreError(f"set_search_params: {ex}"))

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[Vector],
        metadata: list[dict],
    ) -> Result[None, DomainError]:
        """Upsert vectors with metadata into collection.

        Args:
            collection: Collection name
            ids: List of document IDs
            vectors: List of vectors (tuples of floats)
            metadata: List of metadata dicts

        Returns:
            Result with None on success or VectorStoreError on failure
        """
        try:
            models = import_module("qdrant_client.models")

            # Convert vectors to lists (Qdrant expects lists, not tuples)
            vector_lists = [list(v) for v in vectors]

            # Build points with payloads
            points = [
                models.PointStruct(
                    id=ids[i],
                    vector=vector_lists[i],
                    payload=metadata[i],
                )
                for i in range(len(ids))
            ]

            # Batch upsert
            self._client.upsert(
                collection_name=collection,
                points=points,
                wait=True,  # Wait for operation to complete
            )

            return Result.success(None)

        except Exception as ex:
            return Result.failure(VectorStoreError(f"upsert: {ex}"))

    def search(
        self,
        collection: str,
        vector: Vector,
        top_k: int,
        filters: dict | None = None,
    ) -> Result[list[dict], DomainError]:
        """Search for similar vectors in collection.

        Args:
            collection: Collection name
            vector: Query vector (tuple of floats)
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Result with list of dicts [{"id": str, "score": float, "meta": dict}]
            or VectorStoreError on failure
        """
        try:
            models = import_module("qdrant_client.models")

            # Convert vector to list (Qdrant expects lists, not tuples)
            vector_list = list(vector)

            # Build filter if provided
            query_filter = None
            if filters:
                # Simple filter support: {"key": "value"} -> must match
                conditions = [
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                    for k, v in filters.items()
                ]
                query_filter = models.Filter(must=conditions)

            # Search
            results = self._client.search(
                collection_name=collection,
                query_vector=vector_list,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,  # Don't return vectors (saves bandwidth)
            )

            # Convert to standard format
            hits = [
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "meta": dict(hit.payload) if hit.payload else {},
                }
                for hit in results
            ]

            return Result.success(hits)

        except Exception as ex:
            return Result.failure(VectorStoreError(f"search: {ex}"))
