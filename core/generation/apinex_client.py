"""Apinex API client — OpenAI-compatible, used as the primary cloud LLM.

Drop-in replacement surface for both the baseline LLMEngine and the SAH
GroundedGenerator.  The client wraps the official ``openai`` SDK so that
tool-call, streaming, and retry logic are inherited automatically.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

_APINEX_BASE_URL = "https://api.apinex.bond/v1"
_DEFAULT_MODEL = "free/deepseek-v4.1-flash"


class ApinexUnavailable(RuntimeError):
    """Raised when the Apinex API cannot be reached or returns an error."""


class ApinexClient:
    """Thin, production-hardened wrapper around the Apinex OpenAI-compat API.

    Parameters
    ----------
    api_key:
        Apinex API key.  Falls back to the ``APINEX_API_KEY`` environment
        variable when not supplied.
    model:
        Model identifier (e.g. ``"free/deepseek-v4.1-flash"``).  Falls back
        to ``APINEX_MODEL``.
    base_url:
        Apinex base URL.  Override via ``APINEX_BASE_URL`` for staging.
    temperature:
        Sampling temperature forwarded to every generation request.
    max_tokens:
        Upper token budget per completion.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("APINEX_API_KEY", "")
        self.model = model or os.getenv("APINEX_MODEL", _DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("APINEX_BASE_URL", _APINEX_BASE_URL)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Apinex API key is missing.  Set APINEX_API_KEY in your environment."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(self.timeout),
        )

    # ----------------------------------------------------------------- public

    def is_available(self) -> bool:
        """Lightweight availability probe — lists available models."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> str:
        """Send a plain-text prompt and return the model's response text.

        Parameters
        ----------
        prompt:
            User message content.
        temperature:
            Override instance-level temperature for this call.
        max_tokens:
            Override instance-level max_tokens for this call.
        system:
            Optional system message prepended to the conversation.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ApinexUnavailable("Apinex returned an empty response.")
            logger.debug(
                "Apinex (%s) responded with %d chars.", self.model, len(content)
            )
            return content

        except (APIConnectionError, APIError, RateLimitError) as exc:
            raise ApinexUnavailable(f"Apinex API error: {exc}") from exc
        except Exception as exc:
            raise ApinexUnavailable(f"Apinex unexpected error: {exc}") from exc

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a fully-formed messages list (for multi-turn conversations)."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except (APIConnectionError, APIError, RateLimitError) as exc:
            raise ApinexUnavailable(f"Apinex API error: {exc}") from exc


__all__ = ["ApinexClient", "ApinexUnavailable"]

