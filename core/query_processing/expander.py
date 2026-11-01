"""Component 5: Query Expander — Apinex primary, OpenRouter fallback."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand a query into 2-3 semantic variants to improve retrieval recall.

    Provider priority:
      1. Apinex  (APINEX_API_KEY)
      2. OpenRouter (OPEN_ROUTER_API)
      3. No-op  (returns original query only)
    """

    def __init__(self) -> None:
        self._client = None
        self._provider = "none"

        # Tier 1 — Apinex
        apinex_key = os.getenv("APINEX_API_KEY", "").strip()
        if apinex_key:
            try:
                from core.generation.apinex_client import ApinexClient
                self._client = ApinexClient(
                    api_key=apinex_key,
                    model=os.getenv("APINEX_MODEL", "free/deepseek-v4.1-flash"),
                    temperature=0.2,
                    max_tokens=256,
                    timeout=30,
                )
                self._provider = "apinex"
                logger.info("QueryExpander using Apinex (%s).", self._client.model)
                return
            except Exception as exc:
                logger.warning("QueryExpander: Apinex init failed: %s", exc)

        # Tier 2 — OpenRouter
        openrouter_key = os.getenv("OPEN_ROUTER_API", "").strip()
        if openrouter_key:
            try:
                from langchain_openai import ChatOpenAI
                self._client = ChatOpenAI(
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=openrouter_key,
                    model_name=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                    temperature=0.2,
                )
                self._provider = "openrouter"
                logger.info("QueryExpander using OpenRouter fallback.")
            except Exception as exc:
                logger.warning("QueryExpander: OpenRouter init failed: %s", exc)

    # ----------------------------------------------------------------- API

    def expand(self, query: str) -> list[str]:
        """Return the original query plus up to 2 semantic variants."""
        if self._client is None:
            return [query]

        prompt = (
            f"Given the user query: '{query}', generate 2 alternative ways to ask "
            "this question or relevant sub-queries that would help a search engine "
            "find relevant documents. Return ONLY the queries, one per line, no "
            "numbering, no extra text."
        )

        try:
            if self._provider == "apinex":
                raw = self._client.generate(prompt)
            else:
                # LangChain ChatOpenAI interface
                raw = self._client.invoke(prompt).content

            expanded = [q.strip() for q in (raw or "").split("\n") if q.strip()]
            return [query] + expanded[:2]   # original + at most 2 variants

        except Exception as exc:
            logger.warning("Query expansion failed (%s): %s", self._provider, exc)
            return [query]
