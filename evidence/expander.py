"""Context expansion: parent summaries, neighbouring chunks, and siblings."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from evidence.types import EvidenceSet, count_tokens, truncate_tokens
from retrieval.types import Chunk
from sah_config import SAHConfig
from structure.store import StructureStore

logger = logging.getLogger(__name__)


class EvidenceExpander:
    """Pack primary evidence and structure context within a token budget."""

    def __init__(self, store: StructureStore, config: SAHConfig | None = None):
        self.store = store
        self.config = config or SAHConfig.from_env()

    def _add(
        self,
        items: list[Chunk],
        used: set[str],
        chunk: Chunk,
        role: str,
        remaining: int,
        evidence_counter: list[int],
    ) -> int:
        key = f"{chunk.chunk_id}|{role}"
        if key in used or not chunk.text.strip() or remaining <= 0:
            return remaining
        text = truncate_tokens(chunk.text, remaining)
        if not text:
            return remaining
        evidence_counter[0] += 1
        evidence_id = f"E{evidence_counter[0]}"
        item = replace(
            chunk,
            text=text,
            metadata={**chunk.metadata, "role": role, "evidence_id": evidence_id},
        )
        items.append(item)
        used.add(key)
        return remaining - count_tokens(text)

    def expand(self, query: str, chunks: list[Chunk], budget: int | None = None) -> EvidenceSet:
        """Expand primary chunks; context is added only if it fits the budget."""
        budget = budget if budget is not None else self.config.context_token_budget
        items: list[Chunk] = []
        used: set[str] = set()
        counter = [0]
        remaining = budget
        sections: list[str] = []

        # Primary chunks are always first and drive citations.
        for chunk in chunks:
            if chunk.section_id and chunk.section_id not in sections:
                sections.append(chunk.section_id)
            remaining = self._add(items, used, chunk, "primary", remaining, counter)
            if remaining <= 0:
                break

        if self.config.evidence_expansion:
            for primary in chunks:
                if not primary.section_id:
                    continue
                parent = self.store.get_parent(primary.section_id)
                if parent and parent.get("text"):
                    remaining = self._add(
                        items, used,
                        Chunk(
                            text=parent["text"],
                            chunk_id=f"{parent['section_id']}::summary",
                            doc=parent.get("doc") or primary.doc,
                            page=parent.get("page_start"),
                            score=0.0,
                            section_id=parent["section_id"],
                            section_title=parent.get("title"),
                            parent_id=parent.get("parent_id"),
                            page_start=parent.get("page_start"),
                            page_end=parent.get("page_end"),
                            sources=["context"],
                        ),
                        "parent", remaining, counter,
                    )

                for neighbour in self.store.get_neighbour_chunks(primary.chunk_id, radius=1):
                    if neighbour.get("is_context"):
                        continue
                    if neighbour.get("chunk_id") == primary.chunk_id:
                        continue
                    remaining = self._add(
                        items, used,
                        Chunk(
                            text=neighbour.get("text", ""),
                            chunk_id=neighbour.get("chunk_id", ""),
                            doc=neighbour.get("doc") or primary.doc,
                            page=neighbour.get("page_start"),
                            score=0.0,
                            section_id=neighbour.get("section_id"),
                            section_title=neighbour.get("title"),
                            parent_id=neighbour.get("parent_id"),
                            page_start=neighbour.get("page_start"),
                            page_end=neighbour.get("page_end"),
                            sources=["context"],
                        ),
                        "neighbour", remaining, counter,
                    )
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

                for sibling in self.store.get_siblings(primary.section_id)[:3]:
                    remaining = self._add(
                        items, used,
                        Chunk(
                            text=sibling.get("text", ""),
                            chunk_id=f"{sibling['section_id']}::sibling",
                            doc=sibling.get("doc") or primary.doc,
                            page=sibling.get("page_start"),
                            score=0.0,
                            section_id=sibling["section_id"],
                            section_title=sibling.get("title"),
                            parent_id=sibling.get("parent_id"),
                            page_start=sibling.get("page_start"),
                            page_end=sibling.get("page_end"),
                            sources=["context"],
                        ),
                        "sibling", remaining, counter,
                    )
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

        context_lines = []
        template_tokens = 0
        for item in items:
            evidence_id = item.metadata.get("evidence_id", "E?")
            role = item.metadata.get("role", "primary")
            header = (
                f"[{evidence_id}] chunk={item.chunk_id} section={item.section_id} "
                f"title={item.section_title or 'untitled'} page={item.page} role={role}\n"
            )
            template_tokens += count_tokens(header)
            context_lines.append(header + item.text)
        used_tokens = sum(count_tokens(item.text) for item in items) + template_tokens

        logger.info("Evidence expanded: items=%d tokens=%d budget=%d", len(items), used_tokens, budget)
        return EvidenceSet(
            items=items,
            primary=chunks,
            context_text="\n\n".join(context_lines),
            token_count=used_tokens,
            sections=sections,
            truncated=used_tokens >= budget,
        )
