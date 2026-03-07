"""
LLM factory: use Ollama (local) or OpenAI/OpenRouter based on env.
Set USE_OLLAMA=1 or OLLAMA_USE=1 to use local Ollama. OLLAMA_MODEL overrides model name.
"""
from __future__ import annotations

import os
from typing import Any


def use_ollama() -> bool:
    v = os.environ.get("USE_OLLAMA") or os.environ.get("OLLAMA_USE") or ""
    return str(v).lower() in ("1", "true", "yes")


def get_ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2")


def create_llm() -> Any:
    if use_ollama():
        from langchain_ollama import ChatOllama
        return ChatOllama(model=get_ollama_model(), temperature=0)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )
