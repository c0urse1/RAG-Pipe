from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from bu_superagent.application.ports.llm_port import ChatMessage, LLMPort, LLMResponse
from bu_superagent.domain.errors import LLMError

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass
class VLLMOpenAIAdapter(LLMPort):
    base_url: str  # e.g. "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __post_init__(self) -> None:
        # Defer import of OpenAI to chat() to avoid hard dependency in tests
        self._client: Any | None = None

    def chat(
        self, messages: Sequence[ChatMessage], temperature: float = 0.2, max_tokens: int = 512
    ) -> LLMResponse:
        try:
            if self._client is None:
                module = import_module("openai")
                OpenAI = module.OpenAI
                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            assert self._client is not None
            payload: Any = [m.__dict__ for m in messages]
            resp: Any = self._client.chat.completions.create(
                model=self.model,
                messages=cast(Any, payload),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            return LLMResponse(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason,
            )
        except Exception as ex:  # noqa: BLE001
            # Translate external errors to domain-specific errors
            raise LLMError(f"LLM communication failed: {ex}") from ex
