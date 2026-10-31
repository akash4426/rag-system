"""Reciprocal Rank Fusion with chunk deduplication and section expansion."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from retrieval.types import Chunk
from structure.store import StructureStore

RRF_K = 60


def _key(chunk: Chunk) -> str:
    return chunk.chunk_id or hashlib.sha1(chunk.text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def rrf_fuse(
    result_lists: dict[str, list[Chunk]],
    top_k: int = 5,
    weights: dict[str, float] | None = None,
    rrf_k: int = RRF_K,
) -> list[Chunk]:
    """Weighted RRF across named retrievers; duplicates merge their sources."""
    weights = weights or {}
    fused: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    retriever_scores: dict[str, dict[str, float]] = {}

    for retriever, results in result_lists.items():
        weight = float(weights.get(retriever, 1.0))
        for rank, chunk in enumerate(results):
            key = _key(chunk)
            contribution = weight / (rrf_k + rank + 1)
            scores[key] = scores.get(key, 0.0) + contribution
            retriever_scores.setdefault(key, {})[retriever] = chunk.score
            if key not in fused:
                fused[key] = chunk
                fused[key].sources = [retriever]
            else:
                if len(chunk.text) > len(fused[key].text):
                    chunk.chunk_id = fused[key].chunk_id or key
                    chunk.sources = list(fused[key].sources)
                    fused[key] = chunk
                fused[key].sources = sorted(set(fused[key].sources) | {retriever})

    ordered = sorted(fused.values(), key=lambda chunk: scores[_key(chunk)], reverse=True)
    for chunk in ordered:
        key = _key(chunk)
        chunk.score = scores[key]
        chunk.metadata["rrf_score"] = scores[key]
        chunk.metadata["retriever_scores"] = retriever_scores.get(key, {})
    return ordered[:top_k]


def expand_sections(
    store: StructureStore,
    section_ranks: Iterable[tuple[str, int]],
    limit_per_section: int = 20,
) -> list[Chunk]:
    """Expand selected sections into chunks, scored by section rank."""
    ranked = list(section_ranks)
    total = max(len(ranked), 1)
    chunks: list[Chunk] = []
    for rank, (section_id, section_rank) in enumerate(ranked):
        section = store.get_section(section_id)
        if not section:
            continue
        section_chunks = store.get_chunks_for_section(section_id)[:limit_per_section]
        score = (total - rank) / total
        for record in section_chunks:
            chunks.append(
                Chunk(
                    text=record.get("text", ""),
                    chunk_id=record.get("chunk_id", ""),
                    doc=record.get("doc") or "unknown",
                    page=record.get("page_start"),
                    score=score,
                    section_id=section_id,
                    section_title=record.get("title"),
                    parent_id=record.get("parent_id"),
                    page_start=record.get("page_start"),
                    page_end=record.get("page_end"),
                    sources=["structural"],
                    metadata={
                        "section_rank": section_rank,
                        "section_rank_score": score,
                        "structural_rank": rank + 1,
                    },
                )
            )
    return chunks
