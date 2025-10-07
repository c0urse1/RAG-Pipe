"""MinIO blob store adapter for large document storage.

Why: Pipeline-Entkopplung (Backpressure, Retry, Idempotenz) und
     externe Speicherung großer Rohdaten.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from bu_superagent.application.ports.blob_store_port import BlobStorePort
from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result


class BlobStoreError(DomainError):
    """Error in blob storage operations."""


@dataclass
class MinioConfig:
    """Configuration for MinIO client."""

    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str = "rag-documents"
    secure: bool = True
    region: str | None = None


class MinioBlobStoreAdapter(BlobStorePort):
    """MinIO (S3-compatible) adapter for blob storage.

    Features:
    - Store large documents externally (PDFs, Word docs, etc.)
    - Metadata tagging for filtering
    - Presigned URLs for secure access
    - Automatic bucket creation

    Why: External storage for raw documents enables:
         - Vector store focused on vectors (not full text)
         - Document reprocessing without re-upload
         - Cost-effective long-term storage
    """

    def __init__(self, cfg: MinioConfig) -> None:
        """Initialize MinIO blob store adapter.

        Args:
            cfg: MinioConfig with connection parameters

        Raises:
            BlobStoreError: If MinIO initialization fails
        """
        self._cfg = cfg
        self._client = self._init_client(cfg)
        self._ensure_bucket()

    def _init_client(self, cfg: MinioConfig) -> Any:
        """Initialize MinIO client with lazy import.

        Args:
            cfg: MinioConfig with connection parameters

        Returns:
            Minio client instance

        Raises:
            BlobStoreError: If minio-py not available or init fails
        """
        try:
            minio = import_module("minio")
            return minio.Minio(
                cfg.endpoint,
                access_key=cfg.access_key,
                secret_key=cfg.secret_key,
                secure=cfg.secure,
                region=cfg.region,
            )
        except Exception as ex:
            raise BlobStoreError(f"MinIO init failed: {ex}") from ex

    def _ensure_bucket(self) -> None:
        """Ensure bucket exists, create if not.

        Raises:
            BlobStoreError: If bucket creation fails
        """
        try:
            if not self._client.bucket_exists(self._cfg.bucket_name):
                self._client.make_bucket(
                    self._cfg.bucket_name,
                    location=self._cfg.region,
                )
        except Exception as ex:
            raise BlobStoreError(f"Bucket creation failed: {ex}") from ex

    def put(self, key: str, data: bytes, meta: dict[str, Any]) -> Result[str, DomainError]:
        """Put blob data with metadata.

        Args:
            key: Object key (path in bucket)
            data: Binary data to store
            meta: Metadata dict (string keys/values for S3 compatibility)

        Returns:
            Result with storage key or BlobStoreError
        """
        try:
            from io import BytesIO

            # Convert metadata to string values (S3 requirement)
            str_meta = {k: str(v) for k, v in meta.items()}

            # Upload object
            data_stream = BytesIO(data)
            self._client.put_object(
                self._cfg.bucket_name,
                key,
                data_stream,
                length=len(data),
                metadata=str_meta,
            )

            return Result.success(key)

        except Exception as ex:
            return Result.failure(BlobStoreError(f"put failed: {ex}"))

    def get(self, key: str) -> Result[bytes, DomainError]:
        """Get blob data by key.

        Args:
            key: Object key (path in bucket)

        Returns:
            Result with binary data or BlobStoreError
        """
        try:
            # Download object
            response = self._client.get_object(self._cfg.bucket_name, key)
            data = response.read()
            response.close()
            response.release_conn()

            return Result.success(data)

        except Exception as ex:
            return Result.failure(BlobStoreError(f"get failed: {ex}"))

    def get_metadata(self, key: str) -> Result[dict[str, Any], DomainError]:
        """Get object metadata without downloading data.

        Args:
            key: Object key (path in bucket)

        Returns:
            Result with metadata dict or BlobStoreError
        """
        try:
            stat = self._client.stat_object(self._cfg.bucket_name, key)
            return Result.success(dict(stat.metadata) if stat.metadata else {})

        except Exception as ex:
            return Result.failure(BlobStoreError(f"get_metadata failed: {ex}"))

    def delete(self, key: str) -> Result[None, DomainError]:
        """Delete blob by key.

        Args:
            key: Object key (path in bucket)

        Returns:
            Result with None or BlobStoreError
        """
        try:
            self._client.remove_object(self._cfg.bucket_name, key)
            return Result.success(None)

        except Exception as ex:
            return Result.failure(BlobStoreError(f"delete failed: {ex}"))

    def list_keys(self, prefix: str = "") -> Result[list[str], DomainError]:
        """List all object keys with optional prefix.

        Args:
            prefix: Filter keys by prefix (e.g., "docs/2025/")

        Returns:
            Result with list of keys or BlobStoreError
        """
        try:
            objects = self._client.list_objects(
                self._cfg.bucket_name, prefix=prefix, recursive=True
            )
            keys = [obj.object_name for obj in objects]
            return Result.success(keys)

        except Exception as ex:
            return Result.failure(BlobStoreError(f"list_keys failed: {ex}"))
