from typing import Protocol, Optional, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    finish_reason: str = "stop"
    usage_tokens: Optional[int] = None


class LLMPort(Protocol):
    def chat(
        self, messages: Sequence[ChatMessage], temperature: float = 0.2, max_tokens: int = 512
    ) -> LLMResponse: ...
