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

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Convenience method for single-shot text generation.

        Args:
            prompt: The prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string

        Note:
            Default implementation uses chat with a single user message.
            Adapters can override for direct completion APIs.
        """
        msg = ChatMessage(role="user", content=prompt)
        response = self.chat([msg], temperature=temperature, max_tokens=max_tokens)
        return response.text
