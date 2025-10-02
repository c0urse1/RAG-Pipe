"""Deprecated shim.

This module remains for backward-compatibility only. Use
`bu_superagent.infrastructure.llm.vllm_openai_adapter.VLLMOpenAIAdapter` instead.
"""

from .vllm_openai_adapter import VLLMOpenAIAdapter as VLLMOpenAIClient

__all__ = ["VLLMOpenAIClient"]
