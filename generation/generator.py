"""Grounded answer generation over an evidence bundle.

Provider priority (for SAH grounded generation):
  1. Apinex  (cloud, APINEX_API_KEY + SAH_LLM_PROVIDER=apinex or auto)
  2. Ollama  (local, always available as fallback)
"""

from __future__ import annotations

import logging

from evidence.types import EvidenceSet
from generation.prompt import GroundedPromptBuilder
from sah_config import SAHConfig

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """Call the configured generation model with a citation-first evidence prompt.

    When ``SAH_LLM_PROVIDER`` is ``"apinex"`` (or ``APINEX_API_KEY`` is set
    and ``ollama`` is not explicitly forced), Apinex is used as the primary
    cloud generator.  The local Ollama model is always kept as a fallback so
    the system never goes completely dark.
    """

    def __init__(
        self,
        config: SAHConfig | None = None,
        prompt_builder: GroundedPromptBuilder | None = None,
    ) -> None:
        self.config = config or SAHConfig.from_env()
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()

        # ── Tier 1: Apinex ────────────────────────────────────────────────
        self._apinex = None
        if self.config.llm_provider == "apinex" or (
            self.config.llm_provider != "ollama"
            and self.config.apinex_api_key
        ):
            try:
                from core.generation.apinex_client import ApinexClient
                self._apinex = ApinexClient(
                    api_key=self.config.apinex_api_key,
                    model=self.config.apinex_model,
                    temperature=0.0,
                    max_tokens=self.config.apinex_max_tokens,
                    timeout=self.config.apinex_timeout,
                )
                logger.info(
                    "GroundedGenerator: Apinex primary (%s).", self._apinex.model
                )
            except Exception as exc:
                logger.warning(
                    "GroundedGenerator: Apinex init failed, will use Ollama: %s", exc
                )

        # ── Tier 2: Ollama (local fallback, always initialised) ───────────
        from retrieval.ollama_client import OllamaClient
        self._ollama = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            timeout=self.config.ollama_timeout,
        )
        logger.info(
            "GroundedGenerator: Ollama fallback ready (%s).", self._ollama.model
        )

    # ----------------------------------------------------------------- API

    def build_prompt(self, query: str, evidence: EvidenceSet) -> str:
        return self.prompt_builder.build(query, evidence)

    def generate(self, query: str, evidence: EvidenceSet) -> str:
        """Return raw model text; claim splitting / verification is done upstream."""
        prompt = self.build_prompt(query, evidence)
        errors: list[str] = []

        # Try Apinex first
        if self._apinex is not None:
            try:
                from core.generation.apinex_client import ApinexUnavailable
                result = self._apinex.generate(
                    prompt,
                    system=(
                        "You are a grounded question-answering assistant. "
                        "Follow the instructions in the prompt exactly."
                    ),
                    temperature=0.0,
                )
                logger.debug(
                    "GroundedGenerator: Apinex answered (%d chars).", len(result)
                )
                return result
            except Exception as exc:
                errors.append(f"apinex: {exc}")
                logger.warning(
                    "GroundedGenerator: Apinex failed, trying Ollama: %s", exc
                )

        # Fallback to Ollama
        try:
            result = self._ollama.generate(prompt, json_mode=False, temperature=0.0)
            logger.debug(
                "GroundedGenerator: Ollama answered (%d chars).", len(result)
            )
            return result
        except Exception as exc:
            errors.append(f"ollama: {exc}")

        raise RuntimeError(
            "GroundedGenerator: all providers failed — " + " | ".join(errors)
        )
