"""Final-answer generation with a local Ollama fallback.

OpenRouter remains the primary generator. If it is unconfigured, returns an
error, or produces an empty response, generation falls back to the local Ollama
model configured by ``OLLAMA_MODEL`` (``gemma3:4b`` by default).
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

from retrieval.ollama_client import OllamaClient, OllamaUnavailable

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LLMEngine:
    """Component 15: LLM Engine.

    OpenRouter is tried first. When ``OLLAMA_FALLBACK_ENABLED`` is true, a
    failure transparently retries the same prompt against local Ollama.
    """

    def __init__(self):
        api_key = os.getenv("OPEN_ROUTER_API")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.fallback_enabled = _env_bool("OLLAMA_FALLBACK_ENABLED", True)
        if not api_key:
            if self.fallback_enabled:
                print("WARNING: OPEN_ROUTER_API not set. Generation will use the Ollama fallback.")
            else:
                print("WARNING: OPEN_ROUTER_API not set and OLLAMA_FALLBACK_ENABLED is false.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                model_name=model,
                temperature=0.3,
            )

        self.fallback: OllamaClient | None = None
        if self.fallback_enabled:
            try:
                timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
            except (TypeError, ValueError):
                timeout = 120
            self.fallback = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
                timeout=timeout,
            )

    def generate(self, prompt: str) -> str:
        """Return generated text, preserving the original public API."""
        answer, _provider = self.generate_with_provider(prompt)
        return answer

    def generate_with_provider(self, prompt: str) -> tuple[str, str]:
        """Generate text and report which provider actually answered."""
        errors: list[str] = []

        if self.llm is not None:
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                if not str(content or "").strip():
                    raise ValueError("OpenRouter returned an empty response.")
                return str(content), "openrouter"
            except Exception as exc:  # fall through to local generation
                errors.append(f"OpenRouter: {exc}")
                logger.warning("OpenRouter generation failed: %s", exc)
        else:
            errors.append("OpenRouter: OPEN_ROUTER_API is not configured")

        if self.fallback is not None:
            try:
                content = self.fallback.generate(prompt, temperature=0.3)
                if not content.strip():
                    raise OllamaUnavailable("Ollama returned an empty response.")
                provider = f"ollama:{self.fallback.model}"
                logger.warning("Using Ollama fallback model %s.", self.fallback.model)
                return content, provider
            except Exception as exc:
                errors.append(f"Ollama fallback: {exc}")
                logger.error("Ollama fallback failed: %s", exc)

        return "Error during generation: " + " | ".join(errors), "error"


__all__ = ["LLMEngine"]
