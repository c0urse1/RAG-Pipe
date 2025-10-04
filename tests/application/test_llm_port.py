"""Tests for LLM port protocol."""

from collections.abc import Sequence

from bu_superagent.application.ports.llm_port import ChatMessage, LLMPort, LLMResponse


class FakeLLM:
    """Fake LLM adapter for testing."""

    def chat(
        self, messages: Sequence[ChatMessage], temperature: float = 0.2, max_tokens: int = 512
    ) -> LLMResponse:
        # Echo back the last user message
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return LLMResponse(text=f"Echo: {last_user}", finish_reason="stop", usage_tokens=10)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Use default implementation from protocol."""
        msg = ChatMessage(role="user", content=prompt)
        response = self.chat([msg], temperature=temperature, max_tokens=max_tokens)
        return response.text


class TestLLMPort:
    def test_chat_message_creation(self) -> None:
        """ChatMessage should store role and content."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_llm_response_creation(self) -> None:
        """LLMResponse should store text and metadata."""
        resp = LLMResponse(text="Hello!", finish_reason="stop", usage_tokens=5)

        assert resp.text == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage_tokens == 5

    def test_llm_response_defaults(self) -> None:
        """LLMResponse should have sensible defaults."""
        resp = LLMResponse(text="Test")

        assert resp.text == "Test"
        assert resp.finish_reason == "stop"
        assert resp.usage_tokens is None

    def test_fake_llm_implements_protocol(self) -> None:
        """FakeLLM should implement LLMPort protocol."""
        llm: LLMPort = FakeLLM()
        messages = [ChatMessage(role="user", content="Test question")]

        response = llm.chat(messages)

        assert response.text == "Echo: Test question"
        assert response.finish_reason == "stop"

    def test_generate_convenience_method(self) -> None:
        """generate() should provide single-shot text generation."""
        llm: LLMPort = FakeLLM()

        result = llm.generate("What is RAG?")

        assert result == "Echo: What is RAG?"
        assert isinstance(result, str)
