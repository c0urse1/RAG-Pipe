"""Work queue port for async task processing."""

from typing import Any, Protocol

from bu_superagent.domain.errors import DomainError
from bu_superagent.domain.types import Result


class WorkQueuePort(Protocol):
    """Port for async work queue operations."""

    def enqueue(self, topic: str, payload: dict[str, Any]) -> Result[str, DomainError]:
        """Enqueue a task to topic. Returns task ID."""
        ...

    def dequeue_batch(self, topic: str, max_n: int) -> Result[list[dict[str, Any]], DomainError]:
        """Dequeue up to max_n tasks from topic."""
        ...

    def ack(self, topic: str, ack_ids: list[str]) -> Result[None, DomainError]:
        """Acknowledge completion of tasks."""
        ...
