"""Cross-encoder evidence validation.

The generator is never consulted about its own evidence. Relevance and
sufficiency come from a cross-encoder plus transparent term coverage.
"""

from __future__ import annotations

import logging
import math
import re

from evidence.reranker import CrossEncoderReranker
from evidence.types import ValidationResult
from retrieval.types import Chunk
from sah_config import SAHConfig

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "if",
    "in", "into", "is", "it", "its", "of", "on", "or", "should", "that",
    "the", "their", "then", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "document", "documents", "say", "says", "explain", "describe", "give",
    "tell", "according", "section", "chapter", "please", "using", "used",
    "use", "about", "over", "under", "more", "most", "also", "any",
}


def query_aspects(query: str) -> list[str]:
    """Extract light-weight content terms used for coverage checks."""
    tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
    aspects = []
    for token in tokens:
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token not in aspects:
            aspects.append(token)
    return aspects


class EvidenceValidator:
    """Decide sufficiency by comparing the question to retrieved passages."""

    def __init__(
        self,
        reranker: CrossEncoderReranker | None = None,
        config: SAHConfig | None = None,
    ):
        self.reranker = reranker or CrossEncoderReranker()
        self.config = config or SAHConfig.from_env()

    @staticmethod
    def _sigmoid(value: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-value))
        except OverflowError:
            return 0.0 if value < 0 else 1.0

    def validate(
        self,
        query: str,
        chunks: list[Chunk],
        context_text: str = "",
    ) -> ValidationResult:
        """Score relevance/sufficiency and list uncovered query aspects."""
        aspects = query_aspects(query)
        if not chunks:
            return ValidationResult(
                sufficient=False,
                missing_aspects=aspects,
                reasons=["No primary evidence was retrieved."],
                aspects=aspects,
            )

        scores = self.reranker.score(query, [chunk.text for chunk in chunks])
        max_score = max(scores) if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        combined = " ".join(
            [context_text] + [(chunk.section_title or "") for chunk in chunks] + [chunk.text for chunk in chunks]
        ).lower()

        covered = []
        missing = []
        for aspect in aspects:
            if re.search(rf"\b{re.escape(aspect)}\b", combined):
                covered.append(aspect)
            else:
                missing.append(aspect)
        coverage = len(covered) / len(aspects) if aspects else 1.0

        reasons: list[str] = []
        relevance_ok = max_score >= self.config.relevance_threshold
        coverage_ok = coverage >= 0.5
        if not relevance_ok:
            reasons.append(
                f"Max cross-encoder relevance {max_score:.2f} is below threshold "
                f"{self.config.relevance_threshold:.2f}."
            )
        if not coverage_ok:
            reasons.append(
                "Query aspects not covered: " + (", ".join(missing[:6]) if missing else "unknown")
            )
        if relevance_ok and coverage_ok:
            reasons.append(
                f"Max relevance {max_score:.2f} meets threshold; aspect coverage {coverage:.0%}."
            )

        sufficient = bool(chunks) and relevance_ok and coverage_ok
        if not sufficient and missing and "Query aspects not covered" not in " ".join(reasons):
            reasons.append("Uncovered aspects: " + ", ".join(missing[:6]))

        logger.info(
            "Evidence validation: sufficient=%s max=%.2f coverage=%.2f missing=%s",
            sufficient, max_score, coverage, missing[:6],
        )
        return ValidationResult(
            sufficient=sufficient,
            missing_aspects=missing[:6],
            reasons=reasons,
            max_score=max_score,
            mean_score=mean_score,
            coverage=coverage,
            aspects=aspects,
            scores=scores,
        )
