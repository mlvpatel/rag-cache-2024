"""LLM helpers for rag-cache-2024.

The provider is chosen from the model name, so the same code serves OpenAI,
Anthropic, and local Ollama models. Cache augmented generation preloads a whole
corpus into the prompt, so the Ollama context window is widened to fit it.
"""

import logging
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import settings

logger = logging.getLogger(__name__)


def _make_llm(model: str, temperature: float | None = None):
    """Return a chat model for the given model name, provider chosen by name."""
    name = model.lower()
    if any(
        tag in name for tag in ("llama", "qwen", "deepseek", "mistral", "gemma", "phi")
    ):
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": model,
            "base_url": settings.ollama_base_url,
            "num_ctx": settings.ollama_num_ctx,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOllama(**kwargs)
    if "claude" in name:
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)
    from langchain_openai import ChatOpenAI

    kwargs = {"model": model, "api_key": settings.openai_api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def _to_lc_messages(chat_history) -> List[Any]:
    """Convert stored {role, content} dicts into langchain message objects."""
    messages: List[Any] = []
    for turn in chat_history or []:
        if turn.get("role") in ("ai", "assistant"):
            messages.append(AIMessage(content=turn["content"]))
        else:
            messages.append(HumanMessage(content=turn["content"]))
    return messages
