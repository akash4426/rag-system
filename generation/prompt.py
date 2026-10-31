"""Evidence-grounded prompt construction for the generation stage."""

from __future__ import annotations

from evidence.types import EvidenceSet

INSUFFICIENT_SENTINEL = "INSUFFICIENT_EVIDENCE"

SYSTEM_RULES = """You are a grounded question-answering assistant working over indexed documents.

Rules:
1. Use only the evidence passages below. Do not use outside knowledge.
2. Cite every factual sentence with the bracketed evidence IDs that support it,
   for example [E1] or [E1][E3]. Only cite IDs that appear below.
3. Prefer a direct answer in at most four sentences.
4. If the evidence cannot answer the question, reply with exactly
   INSUFFICIENT_EVIDENCE followed by one short reason.
5. Never invent chunk IDs, section IDs, page numbers, or quotations."""


class GroundedPromptBuilder:
    """Build a citation-first prompt from an evidence bundle."""

    def build(self, query: str, evidence: EvidenceSet) -> str:
        context = evidence.context_text.strip() or "(no evidence retrieved)"
        return (
            f"{SYSTEM_RULES}\n\n"
            f"Question: {query.strip()}\n\n"
            f"Evidence passages:\n{context}\n\n"
            "Answer with inline [E#] citations:"
        )
