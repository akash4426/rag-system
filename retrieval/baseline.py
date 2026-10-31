"""Baseline hybrid retrieval exposed through one typed entry point.

This module deliberately wraps the existing BM25 and dense-store components;
it does not replace either retriever.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable

from retrieval.types import Chunk

RRF_K = 60


def text_key(text: str) -> str:
    """Stable fallback identity for corpora that lack chunk IDs."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _page_number(metadata: dict[str, Any]) -> int | None:
    page = metadata.get("page_number")
    if page is not None:
        try:
            return int(page)
        except (TypeError, ValueError):
            pass
    page = metadata.get("page")
    if page is not None:
        try:
            return int(page) + 1  # legacy LangChain PDF pages are 0-based
        except (TypeError, ValueError):
            return None
    return None


def _doc_name(metadata: dict[str, Any]) -> str:
    doc = metadata.get("doc")
    if doc:
        return str(doc)
    source = metadata.get("source") or metadata.get("file_path") or ""
    return os.path.basename(str(source)) if source else "unknown"


class BaselineRetriever:
    """Weighted RRF over the legacy BM25 and dense retrievers."""

    def __init__(
        self,
        tfidf=None,
        vector_db=None,
        embedder=None,
        storage=None,
        structure_store=None,
        rrf_k: int = RRF_K,
    ):
        if tfidf is None or vector_db is None or embedder is None:
            from core.ingestion.embedder import EmbeddingGenerator
            from core.ingestion.storage import Storage
            from core.retrieval.tfidf_retriever import TFIDFRetriever
            from core.retrieval.vector_db_retriever import VectorDBRetriever

            storage = storage if storage is not None else Storage()
            embedder = embedder if embedder is not None else EmbeddingGenerator()
            tfidf = tfidf if tfidf is not None else TFIDFRetriever(storage=storage)
            vector_db = vector_db if vector_db is not None else VectorDBRetriever(storage=storage)

        self.tfidf = tfidf
        self.vector_db = vector_db
        self.embedder = embedder
        self.structure_store = structure_store
        self.rrf_k = rrf_k

    # ---------------------------------------------------------------- helpers
    def _to_chunk(self, item: dict[str, Any], source: str) -> Chunk:
        metadata = dict(item.get("metadata") or {})
        text = item.get("content") or item.get("text") or ""
        chunk_id = item.get("chunk_id") or metadata.get("chunk_id") or text_key(text)
        chunk = Chunk(
            text=text,
            chunk_id=chunk_id,
            doc=_doc_name(metadata),
            page=_page_number(metadata),
            score=float(item.get("score", 0.0)),
            sources=[source],
            metadata=metadata,
        )
        self._enrich(chunk)
        return chunk

    def _enrich(self, chunk: Chunk) -> None:
        """Attach section provenance from SQLite when available."""
        if self.structure_store is None:
            return
        record = None
        try:
            if chunk.chunk_id:
                record = self.structure_store.get_chunk(chunk.chunk_id)
            if record is None and chunk.text:
                record = self.structure_store.find_chunk_by_text(chunk.text)
        except Exception:
            return
        if not record:
            return
        chunk.section_id = record.get("section_id")
        chunk.section_title = record.get("title")
        chunk.parent_id = record.get("parent_id")
        chunk.page_start = record.get("page_start")
        chunk.page_end = record.get("page_end")
        if not chunk.doc or chunk.doc == "unknown":
            chunk.doc = record.get("doc", chunk.doc)

    @staticmethod
    def _fuse(result_lists: Iterable[tuple[str, float, list[Chunk]]], k: int) -> list[Chunk]:
        """Weighted reciprocal rank fusion with chunk deduplication."""
        fused: dict[str, Chunk] = {}
        scores: dict[str, float] = {}

        for source, weight, results in result_lists:
            for rank, chunk in enumerate(results):
                key = chunk.chunk_id or text_key(chunk.text)
                score = weight * (1.0 / (RRF_K + rank + 1))
                scores[key] = scores.get(key, 0.0) + score
                if key not in fused:
                    fused[key] = chunk
                    fused[key].sources = []
                elif len(chunk.text) > len(fused[key].text):
                    # Keep the richer copy, but preserve the stable identity.
                    chunk.chunk_id = fused[key].chunk_id or key
                    fused[key] = chunk
                sources = fused[key].sources
                if source not in sources:
                    sources.append(source)

        ordered = sorted(fused.values(), key=lambda c: scores[c.chunk_id or text_key(c.text)], reverse=True)
        for chunk in ordered:
            key = chunk.chunk_id or text_key(chunk.text)
            chunk.metadata["fused_score"] = scores[key]
            chunk.metadata["rrf_score"] = scores[key]
        return ordered[:k]

    # -------------------------------------------------------------- retrieval
    def retrieve_sparse(self, query: str, k: int = 5) -> list[Chunk]:
        """Return BM25 results as typed chunks."""
        return [self._to_chunk(item, "bm25") for item in self.tfidf.retrieve(query, top_k=k)]

    def retrieve_dense(self, query: str, k: int = 5) -> list[Chunk]:
        """Return dense-store results as typed chunks."""
        embedding = self.embedder.embed_query(query)
        return [self._to_chunk(item, "vector_db") for item in self.vector_db.retrieve(embedding, top_k=k)]

    def retrieve_baseline(self, query: str, k: int = 5) -> list[Chunk]:
        """Run the original hybrid pipeline: BM25 + dense + weighted RRF."""
        pool = max(k * 2, k)
        sparse = self.retrieve_sparse(query, k=pool)
        dense = self.retrieve_dense(query, k=pool)
        return self._fuse([("bm25", 1.0, sparse), ("vector_db", 1.0, dense)], k=k)
