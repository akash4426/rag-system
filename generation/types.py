"""Typed records for grounded generation and citation verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrieval.types import Chunk


@dataclass
class Citation:
    """A generated citation resolved back to a chunk and section."""

    evidence_id: str
    chunk_id: str
    section_id: str | None = None
    section_title: str | None = None
    doc: str = ""
    page: int | None = None
    role: str = "primary"
    text: str = ""

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Citation":
        """Build a citation from an evidence chunk."""
        return cls(
            evidence_id=str(chunk.metadata.get("evidence_id", "")),
            chunk_id=chunk.chunk_id,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            doc=chunk.doc,
            page=chunk.page,
            role=str(chunk.metadata.get("role", "primary")),
            text=chunk.text[:400],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "doc": self.doc,
            "page": self.page,
            "role": self.role,
            "text": self.text,
        }


@dataclass
class Claim:
    """One generated sentence/claim and its citation support."""

    text: str
    citations: list[str] = field(default_factory=list)
    supported: bool = True
    score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": self.citations,
            "supported": self.supported,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass
class VerificationReport:
    """Claim-level citation verification produced without the generator LLM."""

    claims: list[Claim] = field(default_factory=list)
    citation_accuracy: float = 0.0
    mean_score: float = 0.0
    invalid_citations: list[str] = field(default_factory=list)
    missing_citations: int = 0
    verifier: str = "cross-encoder"

    @property
    def claims_total(self) -> int:
        return len(self.claims)

    @property
    def claims_supported(self) -> int:
        return sum(1 for claim in self.claims if claim.supported)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "claims_total": self.claims_total,
            "claims_supported": self.claims_supported,
            "citation_accuracy": self.citation_accuracy,
            "mean_score": self.mean_score,
            "invalid_citations": self.invalid_citations,
            "missing_citations": self.missing_citations,
            "claims": [claim.to_dict() for claim in self.claims],
        }


@dataclass
class GroundedAnswer:
    """Final answer plus citations, verification, and abstention details."""

    query: str
    answer: str
    abstained: bool
    abstention_reason: str = ""
    claims: list[Claim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    supported_parts: list[str] = field(default_factory=list)
    verification: VerificationReport | None = None
    raw_response: str = ""
    generated: bool = True

    def to_dict(self, text_limit: int | None = None) -> dict[str, Any]:
        answer = self.answer
        raw_response = self.raw_response
        if text_limit is not None:
            answer = answer[:text_limit]
            raw_response = raw_response[:text_limit]
        return {
            "query": self.query,
            "answer": answer,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "supported_parts": self.supported_parts,
            "citations": [citation.to_dict() for citation in self.citations],
            "claims": [claim.to_dict() for claim in self.claims],
            "verification": self.verification.to_dict() if self.verification else None,
            "generated": self.generated,
            "raw_response": raw_response,
        }
