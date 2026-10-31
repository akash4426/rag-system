"""Typed evidence records and token helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrieval.types import Chunk

try:
    import tiktoken
except Exception:  # pragma: no cover - tiktoken is an existing dependency
    tiktoken = None


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, falling back to a conservative estimate."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a token budget."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(encoding.encode(text)[:max_tokens])
        except Exception:
            pass
    return text[: max_tokens * 4]


@dataclass
class EvidenceSet:
    """Primary evidence plus context additions, packed to a token budget."""

    items: list[Chunk] = field(default_factory=list)
    primary: list[Chunk] = field(default_factory=list)
    context_text: str = ""
    token_count: int = 0
    sections: list[str] = field(default_factory=list)
    truncated: bool = False

    def trace(self, text_limit: int = 300) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "sections": self.sections,
            "truncated": self.truncated,
            "items": [
                {
                    "evidence_id": item.metadata.get("evidence_id"),
                    "chunk_id": item.chunk_id,
                    "section_id": item.section_id,
                    "section_title": item.section_title,
                    "role": item.metadata.get("role", "primary"),
                    "score": item.score,
                    "text": item.text[:text_limit],
                }
                for item in self.items
            ],
        }


@dataclass
class ValidationResult:
    """Evidence sufficiency produced by a cross-encoder, never by the generator."""

    sufficient: bool
    missing_aspects: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    max_score: float = 0.0
    mean_score: float = 0.0
    coverage: float = 0.0
    aspects: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "missing_aspects": self.missing_aspects,
            "reasons": self.reasons,
            "max_score": self.max_score,
            "mean_score": self.mean_score,
            "coverage": self.coverage,
            "aspects": self.aspects,
        }
