"""Rule-based query analysis for adaptive retrieval.

No learned classifier is used. Signals are intentionally simple and logged so
each retrieval decision can be inspected.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_COMPARE = {"compare", "comparison", "difference", "differences", "differ", "versus", "vs", "contrast"}
_STRUCTURE = {"section", "chapter", "part", "heading", "toc", "appendix", "table of contents"}
_EXPLAIN = {"explain", "summarize", "summary", "overview", "describe", "analysis", "why"}
_FACTUAL = {"what", "who", "when", "where", "which", "how", "how many"}


@dataclass
class ControllerDecision:
    """What to run, how deep, and why."""

    retrievers: list[str]
    k: int
    complexity: str
    reason: str
    signals: dict[str, object] = field(default_factory=dict)


class QueryController:
    """Cheap signal-driven controller for semantic/lexical/structural routing."""

    def analyze(self, query: str, base_k: int = 5) -> ControllerDecision:
        text = (query or "").strip().lower()
        words = re.findall(r"[a-z0-9]+", text)
        word_count = len(words)
        compare_hits = sorted(_COMPARE & set(words))
        structure_hits = sorted(word for word in _STRUCTURE if word in text)
        explain_hits = sorted(_EXPLAIN & set(words))
        conjunction_count = text.count(" and ")
        comma_count = text.count(",")
        factual = any(word in words for word in _FACTUAL)
        multi_entity = conjunction_count > 0 or comma_count > 0

        signals = {
            "word_count": word_count,
            "compare_terms": compare_hits,
            "structure_terms": structure_hits,
            "explain_terms": explain_hits,
            "conjunction_count": conjunction_count,
            "comma_count": comma_count,
            "factual": factual,
            "multi_entity": multi_entity,
        }

        if compare_hits or (multi_entity and word_count >= 10):
            retrievers = ["semantic", "lexical", "structural"]
            k = min(max(base_k * 2, 8), 12)
            complexity = "complex"
            reason = (
                "Comparison/multi-entity signal: run semantic and lexical for coverage "
                "plus structural for section alignment; widen retrieval depth."
            )
        elif structure_hits:
            retrievers = ["structural", "semantic"]
            k = max(base_k, 5)
            complexity = "moderate"
            reason = "Explicit section/chapter reference: structural routing first, semantic fallback second."
        elif word_count >= 28 or len(explain_hits) >= 2:
            retrievers = ["semantic", "lexical", "structural"]
            k = min(max(base_k * 2, 8), 12)
            complexity = "complex"
            reason = "Long explanatory query: combine all three retrievers and widen depth."
        elif word_count <= 8 and factual:
            retrievers = ["semantic", "lexical"]
            k = base_k
            complexity = "simple"
            reason = "Short factual query: semantic + lexical retrieval at baseline depth."
        else:
            retrievers = ["semantic", "lexical"]
            k = base_k
            complexity = "moderate"
            reason = "Default: semantic + lexical hybrid retrieval at baseline depth."

        decision = ControllerDecision(
            retrievers=retrievers, k=k, complexity=complexity, reason=reason, signals=signals
        )
        logger.info(
            "Controller: retrievers=%s k=%d complexity=%s reason=%s",
            decision.retrievers, decision.k, decision.complexity, decision.reason,
        )
        return decision
