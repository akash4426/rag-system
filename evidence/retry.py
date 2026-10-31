"""Structure-guided retry and the plain re-query ablation."""

from __future__ import annotations

import logging

from evidence.reranker import CrossEncoderReranker
from retrieval.types import Chunk
from sah_config import SAHConfig
from structure.store import StructureStore

logger = logging.getLogger(__name__)


class EvidenceRetrier:
    """Target missing aspects in the neighbourhood of already-hit sections."""

    def __init__(
        self,
        store: StructureStore,
        reranker: CrossEncoderReranker,
        config: SAHConfig | None = None,
    ):
        self.store = store
        self.reranker = reranker
        self.config = config or SAHConfig.from_env()

    def _to_chunk(self, record: dict) -> Chunk:
        return Chunk(
            text=record.get("text", ""),
            chunk_id=record.get("chunk_id", ""),
            doc=record.get("doc") or "unknown",
            page=record.get("page_start"),
            score=0.0,
            section_id=record.get("section_id"),
            section_title=record.get("title"),
            parent_id=record.get("parent_id"),
            page_start=record.get("page_start"),
            page_end=record.get("page_end"),
            sources=["structural_retry"],
            metadata={"retry_source": "neighbourhood"},
        )

    def structure_guided(
        self,
        query: str,
        missing_aspects: list[str],
        hit_chunks: list[Chunk],
        k: int = 5,
        attempt: int = 1,
    ) -> tuple[list[Chunk], dict]:
        """Retrieve parent/sibling/adjacent sections around hit sections."""
        section_ids = []
        for chunk in hit_chunks:
            if chunk.section_id and chunk.section_id not in section_ids:
                section_ids.append(chunk.section_id)
        if not section_ids:
            return [], {"mode": "structure", "reason": "No hit sections to expand."}

        neighbourhood = self.store.get_neighbourhood(section_ids[0], adjacent=attempt)
        seen_sections = {item["section_id"] for item in neighbourhood}
        for section_id in section_ids[1:]:
            for item in self.store.get_neighbourhood(section_id, adjacent=attempt):
                if item["section_id"] not in seen_sections:
                    neighbourhood.append(item)
                    seen_sections.add(item["section_id"])

        existing = {chunk.chunk_id for chunk in hit_chunks}
        records = self.store.get_chunks_for_sections([item["section_id"] for item in neighbourhood])
        candidates = [self._to_chunk(record) for record in records if record.get("chunk_id") not in existing]
        if not candidates:
            return [], {"mode": "structure", "reason": "Neighbourhood had no new chunks.", "sections": list(seen_sections)}

        targeted_query = query
        if missing_aspects:
            targeted_query = f"{query} {' '.join(missing_aspects)}"
        candidates = self.reranker.rerank(targeted_query, candidates, top_k=k)
        event = {
            "mode": "structure",
            "query": targeted_query,
            "sections": sorted(section_ids),
            "neighbourhood": sorted(seen_sections),
            "new_chunk_ids": [chunk.chunk_id for chunk in candidates],
            "reason": "Expanded parent/sibling/adjacent sections around hits.",
        }
        logger.info("Structure-guided retry: sections=%d new_chunks=%d", len(seen_sections), len(candidates))
        return candidates, event

    def plain(
        self,
        query: str,
        missing_aspects: list[str],
        adaptive,
        k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[Chunk], dict]:
        """Ablation: generic re-query with the missing aspects appended."""
        targeted_query = query
        if missing_aspects:
            targeted_query = f"{query} {' '.join(missing_aspects)}"
        result = adaptive.retrieve_adaptive(targeted_query, k=k, doc_ids=doc_ids)
        chunks = result.chunks
        event = {
            "mode": "plain",
            "query": targeted_query,
            "new_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "reason": "Plain generic re-query with missing aspects.",
        }
        logger.info("Plain retry: chunks=%d", len(chunks))
        return chunks, event
