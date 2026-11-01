"""Final-answer generation with a three-tier provider chain.

Priority order:
  1. Apinex  (primary cloud LLM — free/deepseek-v4.1-flash)
  2. OpenRouter (secondary cloud LLM — gpt-4o-mini)
  3. Ollama  (local fallback — gemma3:4b)

Each provider is tried in sequence; the first to return a non-empty response
wins.  Failures are logged but never surfaced to the caller as exceptions.
"""

from __future__ import annotations

import logging
import os

from core.generation.apinex_client import ApinexClient, ApinexUnavailable
from retrieval.ollama_client import OllamaClient, OllamaUnavailable

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class LLMEngine:
    """Component 15: LLM Engine — three-tier provider chain.

    Apinex is the default cloud provider for production.  OpenRouter and the
    local Ollama model are retained as automatic fallbacks so the system
    degrades gracefully when any single provider is unavailable.
    """

    def __init__(self) -> None:
        # ── Tier 1: Apinex (primary) ──────────────────────────────────────
        self.apinex: ApinexClient | None = None
        apinex_key = os.getenv("APINEX_API_KEY", "").strip()
        if apinex_key:
            try:
                self.apinex = ApinexClient(
                    api_key=apinex_key,
                    model=os.getenv("APINEX_MODEL", "free/deepseek-v4.1-flash"),
                    temperature=0.2,
                    max_tokens=_env_int("APINEX_MAX_TOKENS", 1024),
                    timeout=_env_int("APINEX_TIMEOUT", 60),
                )
                logger.info("Apinex client initialised (model: %s).", self.apinex.model)
            except Exception as exc:
                logger.warning("Apinex client failed to initialise: %s", exc)
        else:
            logger.warning("APINEX_API_KEY not set — Apinex provider disabled.")

        # ── Tier 2: OpenRouter (secondary) ────────────────────────────────
        self.openrouter_llm = None
        openrouter_key = os.getenv("OPEN_ROUTER_API", "").strip()
        if openrouter_key:
            try:
                from langchain_openai import ChatOpenAI
                self.openrouter_llm = ChatOpenAI(
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=openrouter_key,
                    model_name=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                    temperature=0.3,
                )
                logger.info("OpenRouter client initialised as secondary fallback.")
            except Exception as exc:
                logger.warning("OpenRouter client failed to initialise: %s", exc)
        else:
            logger.info("OPEN_ROUTER_API not set — OpenRouter provider disabled.")

        # ── Tier 3: Ollama (local last-resort) ────────────────────────────
        self.ollama: OllamaClient | None = None
        if _env_bool("OLLAMA_FALLBACK_ENABLED", True):
            self.ollama = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
                timeout=_env_int("OLLAMA_TIMEOUT", 120),
            )
            logger.info("Ollama local fallback ready (model: %s).", self.ollama.model)

    # ----------------------------------------------------------------- API

    def generate(self, prompt: str) -> str:
        """Return generated text (public API, preserved for backward compat)."""
        answer, _provider = self.generate_with_provider(prompt)
        return answer

    def generate_with_provider(self, prompt: str) -> tuple[str, str]:
        """Generate text and report which provider actually answered.

        Returns
        -------
        (text, provider_label)
            ``provider_label`` is one of ``"apinex"``, ``"openrouter"``,
            ``"ollama:<model>"``, or ``"error"``.
        """
        errors: list[str] = []

        # Tier 1 — Apinex
        if self.apinex is not None:
            try:
                content = self.apinex.generate(prompt)
                logger.debug("Apinex answered (%d chars).", len(content))
                return content, f"apinex:{self.apinex.model}"
            except ApinexUnavailable as exc:
                errors.append(f"Apinex: {exc}")
                logger.warning("Apinex failed, trying next provider: %s", exc)

        # Tier 2 — OpenRouter
        if self.openrouter_llm is not None:
            try:
                response = self.openrouter_llm.invoke(prompt)
                content = str(response.content or "").strip()
                if not content:
                    raise ValueError("OpenRouter returned an empty response.")
                logger.warning("Using OpenRouter fallback.")
                return content, "openrouter"
            except Exception as exc:
                errors.append(f"OpenRouter: {exc}")
                logger.warning("OpenRouter failed, trying Ollama: %s", exc)
        else:
            errors.append("OpenRouter: not configured")

        # Tier 3 — Ollama
        if self.ollama is not None:
            try:
                content = self.ollama.generate(prompt, temperature=0.3)
                if not content.strip():
                    raise OllamaUnavailable("Ollama returned an empty response.")
                provider = f"ollama:{self.ollama.model}"
                logger.warning("Using Ollama local fallback (%s).", self.ollama.model)
                return content, provider
            except Exception as exc:
                errors.append(f"Ollama: {exc}")
                logger.error("Ollama fallback also failed: %s", exc)

        error_msg = "Error: all LLM providers failed — " + " | ".join(errors)
        logger.error(error_msg)
        return error_msg, "error"


__all__ = ["LLMEngine"]
