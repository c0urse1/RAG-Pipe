from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from bu_superagent.application.ports.llm_port import LLMPort, ChatMessage, LLMResponse


@dataclass
class VLLMOpenAIAdapter(LLMPort):
    base_url: str  # e.g. "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self, messages: Sequence[ChatMessage], temperature: float = 0.2, max_tokens: int = 512
    ) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[m.__dict__ for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            return LLMResponse(text=choice.message.content or "", finish_reason=choice.finish_reason)
        except Exception as ex:  # noqa: BLE001
            # Übersetze externe Fehler in domänenspezifische – hier kurz gehalten:
            raise RuntimeError(f"LLM communication failed: {ex}")  # später: typisierte Errors
