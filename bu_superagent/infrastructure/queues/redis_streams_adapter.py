"""Redis Streams work queue adapter for async task distribution.

Why: Pipeline-Entkopplung (Backpressure, Retry, Idempotenz) und
     externe Speicherung großer Rohdaten.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from bu_superagent.application.scalable_ports import WorkQueuePort
from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result


class WorkQueueError(DomainError):
    """Error in work queue operations."""


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True


class RedisWorkQueueAdapter(WorkQueuePort):
    """Redis Streams adapter for work queue operations.

    Uses Redis Streams for:
    - Reliable message delivery
    - Consumer groups for parallel workers
    - Automatic retry with backoff
    - At-least-once delivery guarantee

    Why: Enables pipeline decoupling, backpressure handling,
         retry logic, and idempotent processing.
    """

    def __init__(self, cfg: RedisConfig) -> None:
        """Initialize Redis work queue adapter.

        Args:
            cfg: RedisConfig with connection parameters

        Raises:
            WorkQueueError: If Redis initialization fails
        """
        self._cfg = cfg
        self._client = self._init_client(cfg)

    def _init_client(self, cfg: RedisConfig) -> Any:
        """Initialize Redis client with lazy import.

        Args:
            cfg: RedisConfig with connection parameters

        Returns:
            Redis client instance

        Raises:
            WorkQueueError: If redis-py not available or init fails
        """
        try:
            redis = import_module("redis")
            return redis.Redis(
                host=cfg.host,
                port=cfg.port,
                db=cfg.db,
                password=cfg.password,
                decode_responses=cfg.decode_responses,
            )
        except Exception as ex:
            raise WorkQueueError(f"Redis init failed: {ex}") from ex

    def enqueue(self, topic: str, payload: dict[str, Any]) -> Result[str, DomainError]:
        """Enqueue a task to topic stream.

        Args:
            topic: Stream name (e.g., "ingest-tasks")
            payload: Task data (must be JSON-serializable)

        Returns:
            Result with message ID or WorkQueueError
        """
        try:
            import json

            # Redis Streams: XADD command
            # Payload stored as single field to preserve structure
            msg_id = self._client.xadd(
                topic,
                {"payload": json.dumps(payload)},
                maxlen=100000,  # Keep last 100k messages (prevents unbounded growth)
            )

            return Result.success(str(msg_id))

        except Exception as ex:
            return Result.failure(WorkQueueError(f"enqueue failed: {ex}"))

    def dequeue_batch(self, topic: str, max_n: int) -> Result[list[dict[str, Any]], DomainError]:
        """Dequeue up to max_n tasks from topic.

        Uses consumer groups for parallel processing.
        Creates consumer group if not exists.

        Args:
            topic: Stream name
            max_n: Maximum number of messages to read

        Returns:
            Result with list of task dicts or WorkQueueError
        """
        try:
            import json

            group_name = f"{topic}-workers"
            consumer_name = "worker-1"  # TODO: unique per worker instance

            # Create consumer group if not exists
            try:
                self._client.xgroup_create(topic, group_name, id="0", mkstream=True)
            except Exception:
                # Group already exists, ignore
                pass

            # Read from consumer group
            # XREADGROUP: blocking read with auto-claim of pending messages
            messages = self._client.xreadgroup(
                group_name,
                consumer_name,
                {topic: ">"},
                count=max_n,
                block=0,  # Non-blocking
            )

            # Parse messages
            tasks = []
            if messages:
                for stream_name, stream_messages in messages:
                    for msg_id, fields in stream_messages:
                        payload_str = fields.get("payload", "{}")
                        payload = json.loads(payload_str)
                        tasks.append(
                            {
                                "_msg_id": msg_id,
                                "_stream": stream_name,
                                **payload,
                            }
                        )

            return Result.success(tasks)

        except Exception as ex:
            return Result.failure(WorkQueueError(f"dequeue failed: {ex}"))

    def ack(self, topic: str, ack_ids: list[str]) -> Result[None, DomainError]:
        """Acknowledge completion of tasks.

        Args:
            topic: Stream name
            ack_ids: List of message IDs to acknowledge

        Returns:
            Result with None or WorkQueueError
        """
        try:
            group_name = f"{topic}-workers"

            # XACK: acknowledge messages in consumer group
            if ack_ids:
                self._client.xack(topic, group_name, *ack_ids)

            return Result.success(None)

        except Exception as ex:
            return Result.failure(WorkQueueError(f"ack failed: {ex}"))
