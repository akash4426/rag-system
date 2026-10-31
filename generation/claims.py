"""Rule-based claim and citation parsing for grounded answers."""

from __future__ import annotations

import re

_CITATION_BLOCK = re.compile(r"\[((?:\s*E\d+\s*[,;]?\s*)+)\]", re.IGNORECASE)
_CITATION_TOKEN = re.compile(r"E(\d+)", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def extract_citation_ids(text: str) -> list[str]:
    """Return unique evidence IDs in first-seen order, e.g. ``[E1, E3]``."""
    found: list[str] = []
    for block in _CITATION_BLOCK.findall(text or ""):
        for number in _CITATION_TOKEN.findall(block):
            evidence_id = f"E{int(number)}"
            if evidence_id not in found:
                found.append(evidence_id)
    return found


def strip_citation_markers(text: str) -> str:
    """Remove evidence markers while leaving the claim text intact."""
    return re.sub(r"\s*\[(?:E\d+[\s,;]*)\]", "", text or "").strip()


def split_claims(text: str) -> list[str]:
    """Split an answer into sentence-level claims with a transparent rule set.

    Bullets and line breaks are hard boundaries; within a paragraph, sentence
    punctuation followed by a capital letter, digit, or citation marker splits.
    """
    claims: list[str] = []
    for raw_line in (text or "").splitlines():
        line = _BULLET.sub("", raw_line).strip()
        if not line:
            continue
        for part in _SENTENCE_SPLIT.split(line):
            claim = part.strip()
            if len(claim) < 3:
                continue
            if not re.search(r"[A-Za-z]", claim):
                continue
            claims.append(claim)
    return claims


def split_claim_records(text: str) -> list[tuple[str, list[str]]]:
    """Return ``(claim_text, citation_ids)`` pairs for an answer."""
    return [(claim, extract_citation_ids(claim)) for claim in split_claims(text)]
