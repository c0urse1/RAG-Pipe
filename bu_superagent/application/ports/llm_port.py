from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    finish_reason: str = "stop"
    usage_tokens: int | None = None


class LLMPort(Protocol):
    def chat(
        self, messages: Sequence[ChatMessage], temperature: float = 0.2, max_tokens: int = 512
    ) -> LLMResponse: ...
