"""Response-shaping helpers for retrievable citations and frontend traces."""

from __future__ import annotations

import hashlib
from typing import Any

from evidence.types import EvidenceSet
from retrieval.types import Chunk


def _doc_name(metadata: dict[str, Any]) -> str:
    doc = metadata.get("doc") or metadata.get("source") or metadata.get("file_path") or ""
    return str(doc).replace("\\", "/").rsplit("/", 1)[-1] if doc else "unknown"


def _page(metadata: dict[str, Any]) -> int | None:
    page = metadata.get("page_number", metadata.get("page"))
    if page is None:
        return None
    try:
        return int(page) + (1 if "page" in metadata and "page_number" not in metadata else 0)
    except (TypeError, ValueError):
        return None


def citation_from_chunk(chunk: Chunk, index: int) -> dict[str, Any]:
    """Serialize an evidence chunk as a structured citation."""
    evidence_id = chunk.metadata.get("evidence_id") or f"S{index}"
    return {
        "evidence_id": str(evidence_id),
        "chunk_id": chunk.chunk_id,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "doc": chunk.doc,
        "page": chunk.page,
        "role": chunk.metadata.get("role", "primary"),
        "score": chunk.score,
        "text": chunk.text[:400],
    }


def citation_from_legacy(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Serialize a legacy reranker result using the same citation shape."""
    metadata = dict(item.get("metadata") or {})
    text = str(item.get("content") or item.get("text") or "")
    chunk_id = str(metadata.get("chunk_id") or hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
    return {
        "evidence_id": f"S{index}",
        "chunk_id": chunk_id,
        "section_id": metadata.get("section_id"),
        "section_title": metadata.get("section_title") or metadata.get("title"),
        "doc": _doc_name(metadata),
        "page": _page(metadata),
        "role": "primary",
        "score": item.get("rerank_score", item.get("score", 0.0)),
        "text": text[:400],
    }


def context_chunk_from_citation(citation: dict[str, Any]) -> dict[str, Any]:
    """Convert a citation to the legacy frontend context-chunk shape."""
    source = citation.get("doc") or "unknown"
    if citation.get("section_title"):
        source = f"{source} · {citation['section_title']}"
    return {"source": source, **citation}


def evidence_context_chunks(evidence: EvidenceSet) -> list[dict[str, Any]]:
    """Return every expanded evidence item for the source explorer."""
    return [
        context_chunk_from_citation(citation_from_chunk(item, index + 1))
        for index, item in enumerate(evidence.items)
    ]


def legacy_context_chunks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return legacy reranker results in the shared context shape."""
    return [
        context_chunk_from_citation(citation_from_legacy(item, index + 1))
        for index, item in enumerate(results)
    ]


def legacy_trace(
    intent: str,
    expanded_queries: list[str],
    results: list[dict[str, Any]],
    sparse_weight: float,
    dense_weight: float,
    candidate_count: int | None = None,
    provider: str = "openrouter",
) -> dict[str, Any]:
    """Trace for the untouched baseline path."""
    return {
        "mode": "baseline",
        "query_processing": {"intent": intent, "expanded_queries": expanded_queries},
        "retrieval": {
            "retrievers": ["bm25", "vector_db"],
            "weights": {"sparse": sparse_weight, "dense": dense_weight},
            "candidates": candidate_count if candidate_count is not None else len(results),
            "returned": len(results),
            "chunks": [citation_from_legacy(item, index + 1) for index, item in enumerate(results)],
        },
        "evidence": None,
        "validation": None,
        "retries": [],
        "generation": {"provider": provider, "grounded": False},
    }
