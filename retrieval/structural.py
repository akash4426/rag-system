"""STAIR-inspired structural retrieval.

Given a compact table of contents, ask the local Ollama model to choose
relevant leaf sections, validate the IDs, then expand those sections into
chunks. This is inspired by the idea of LLM-guided section selection; it is
not a reproduction of the STAIR paper or its implementation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from retrieval.ollama_client import OllamaClient, OllamaUnavailable, parse_json_payload
from retrieval.types import Chunk
from structure.store import StructureStore
from sah_config import SAHConfig
from utils.progress import ProgressCallback, emit_progress

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "does", "do",
}


@dataclass
class StructuralResult:
    """Selected sections plus expanded chunks and a small audit trail."""

    chunks: list[Chunk]
    selected_sections: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = "llm"
    raw_response: str = ""
    retries: int = 0
    reason: str = ""


class StructuralRetriever:
    """LLM-selects leaf sections from a ToC, then expands to section chunks."""

    def __init__(
        self,
        store: StructureStore,
        llm: OllamaClient | None = None,
        config: SAHConfig | None = None,
    ):
        self.store = store
        self.config = config or SAHConfig.from_env()
        self.llm = llm or OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            timeout=self.config.ollama_timeout,
        )
        self._available: bool | None = None
        self._available_checked_at: float = 0.0
        self._max_toc_sections = 150
        # Re-check Ollama availability every 30 seconds after a failure so
        # the system auto-recovers when Ollama comes back up.
        self._recheck_interval: float = 30.0

    # ---------------------------------------------------------------- ToC view
    def _documents(self, doc_ids: list[str] | None) -> list[dict[str, Any]]:
        if doc_ids:
            documents = [self.store.get_document(doc_id) for doc_id in doc_ids]
            return [doc for doc in documents if doc]
        return self.store.list_documents()

    def _render_toc(self, documents: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
        """Build a compact hierarchical ToC and an ID -> section map."""
        lines: list[str] = []
        leaf_map: dict[str, dict[str, Any]] = {}
        rendered = 0
        for document in documents:
            sections = self.store.get_sections(document["doc_id"])
            if not sections:
                continue
            leaves = self.store.get_leaf_sections(document["doc_id"])
            leaf_ids = {section["section_id"] for section in leaves}
            for leaf in leaves:
                leaf_map[leaf["section_id"]] = {**leaf, "doc": document["filename"]}
            lines.append(f"\nDOCUMENT {document['doc_id']}: {document['filename']}")
            for section in sections:
                if rendered >= self._max_toc_sections:
                    lines.append("  ... table of contents truncated ...")
                    break
                indent = "  " * max(0, int(section.get("level", 1)) - 1)
                marker = "  [leaf]" if section["section_id"] in leaf_ids else ""
                lines.append(f"{indent}[{section['section_id']}] {section['title']}{marker}")
                rendered += 1
        return "\n".join(lines).strip(), leaf_map

    # ------------------------------------------------------------- LLM prompt
    @staticmethod
    def _build_prompt(query: str, toc: str, top_n: int) -> str:
        return f"""/no_think
You navigate a document by its table of contents.

Select up to {top_n} leaf sections that are most likely to contain the answer.
Use section titles and hierarchy only. Prefer specific sections over broad ones.
Return JSON only, exactly this shape:
{{"section_ids": ["<id>", "<id>"]}}

Rules:
- Use IDs copied exactly from the table of contents.
- Choose only entries marked [leaf].
- Do not invent IDs.
- Return an empty list if nothing is relevant.
- No prose, no Markdown.

QUESTION:
{query}

TABLE OF CONTENTS:
{toc}
"""

    def _parse_ids(self, text: str, valid_ids: set[str], top_n: int) -> tuple[list[str], bool]:
        """Return (valid IDs, parsed_ok). Valid IDs do not imply relevance."""
        try:
            payload = parse_json_payload(text)
        except ValueError:
            return [], False

        values: list[Any] = []
        if isinstance(payload, dict):
            for key in ("section_ids", "sections", "ids", "selected_sections"):
                if key in payload:
                    values = payload[key] if isinstance(payload[key], list) else [payload[key]]
                    break
        elif isinstance(payload, list):
            values = payload

        selected: list[str] = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("section_id") or value.get("id")
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value in valid_ids and value not in selected:
                selected.append(value)
        return selected[:top_n], True

    def _keyword_fallback(
        self, query: str, leaf_map: dict[str, dict[str, Any]], top_n: int
    ) -> list[str]:
        """Deterministic fallback when Ollama is unavailable."""
        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term not in _STOPWORDS]
        scored: list[tuple[float, str]] = []
        for section_id, section in leaf_map.items():
            title = (section.get("title") or "").lower()
            body = (section.get("text") or "").lower()[:400]
            score = sum(3.0 for term in terms if term in title)
            score += sum(0.5 for term in terms if term in body)
            if score > 0:
                scored.append((score, section_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if scored:
            return [section_id for _, section_id in scored[:top_n]]
        return list(leaf_map)[:top_n]

    @staticmethod
    def _selection_event(
        callback: ProgressCallback | None,
        selected_ids: list[str],
        leaf_map: dict[str, dict[str, Any]],
        info: dict[str, Any],
        model: str,
    ) -> None:
        sections = [
            {
                "section_id": section_id,
                "title": leaf_map.get(section_id, {}).get("title"),
                "doc": leaf_map.get(section_id, {}).get("doc"),
            }
            for section_id in selected_ids
        ]
        strategy = info.get("strategy", "unknown")
        emit_progress(
            callback,
            "structural_select",
            "done" if strategy == "llm" else "warning",
            (
                f"STAIR-inspired selector chose {len(sections)} section(s) with {model}."
                if strategy == "llm"
                else f"Structural selector used {strategy}: {len(sections)} section(s)."
            ),
            strategy=strategy,
            selected_sections=sections,
            retries=int(info.get("retries", 0)),
            reason=info.get("reason", ""),
            model=model,
        )

    # --------------------------------------------------------------- selection
    def select_sections(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_n: int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Choose leaf section IDs, validating every model output."""
        top_n = top_n or self.config.structural_top_n
        documents = self._documents(doc_ids)
        toc, leaf_map = self._render_toc(documents)
        if not leaf_map:
            emit_progress(
                on_event,
                "structural_select",
                "error",
                "No stored sections are available for the requested documents.",
                documents=[document.get("doc_id") for document in documents],
            )
            return [], {"strategy": "no_sections", "reason": "No stored sections for the requested documents."}

        emit_progress(
            on_event,
            "structural_select",
            "running",
            f"STAIR-inspired selector is reading {len(leaf_map)} leaf sections.",
            documents=[document.get("doc_id") for document in documents],
            leaf_sections=len(leaf_map),
            top_n=top_n,
            model=self.config.ollama_model,
        )

        prompt = self._build_prompt(query, toc, top_n)
        valid_ids = set(leaf_map)
        attempts = 0
        raw = ""
        parsed_ok = False

        if self._available is None or (
            self._available is False
            and (time.time() - self._available_checked_at) >= self._recheck_interval
        ):
            self._available = self.llm.is_available()
            self._available_checked_at = time.time()
            if self._available:
                logger.info("Ollama is now available for structural selection.")
        if self._available:
            for attempt in range(2):
                attempts = attempt
                try:
                    raw = self.llm.generate(prompt, json_mode=True, temperature=0.0)
                except OllamaUnavailable as exc:
                    logger.warning("Structural selector unavailable: %s", exc)
                    self._available = False
                    self._available_checked_at = time.time()
                    break
                selected, parsed_ok = self._parse_ids(raw, valid_ids, top_n)
                if parsed_ok:
                    if selected:
                        info = {
                            "strategy": "llm",
                            "reason": f"LLM selected {len(selected)} valid leaf section(s).",
                            "retries": attempts,
                            "raw_response": raw[:2000],
                            "documents": [doc["doc_id"] for doc in documents],
                        }
                        self._selection_event(on_event, selected, leaf_map, info, self.config.ollama_model)
                        return selected, info
                    break
                logger.warning("Malformed structural JSON (attempt %d); retrying once.", attempt + 1)
            info = {
                "strategy": "keyword_fallback",
                "reason": (
                    "LLM returned no valid IDs; used keyword fallback. "
                    "Valid IDs do not guarantee relevance."
                ),
                "retries": attempts,
                "raw_response": raw[:2000],
                "documents": [doc["doc_id"] for doc in documents],
            }
        else:
            info = {
                "strategy": "keyword_fallback",
                "reason": "Ollama unavailable; used deterministic keyword fallback over section titles.",
                "retries": 0,
                "raw_response": "",
                "documents": [doc["doc_id"] for doc in documents],
            }

        selected = self._keyword_fallback(query, leaf_map, top_n)
        info["selected_sections"] = selected
        self._selection_event(on_event, selected, leaf_map, info, self.config.ollama_model)
        return selected, info

    # --------------------------------------------------------------- expansion
    def retrieve_structural(
        self,
        query: str,
        k: int = 5,
        doc_ids: list[str] | None = None,
        top_n: int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> StructuralResult:
        """Select sections and expand their chunks with rank-based scores."""
        selected_ids, info = self.select_sections(
            query, doc_ids=doc_ids, top_n=top_n, on_event=on_event
        )
        sections = [self.store.get_section(section_id) for section_id in selected_ids]
        sections = [section for section in sections if section]
        selected_meta: list[dict[str, Any]] = []
        rank_by_id: dict[str, int] = {}

        for rank, section in enumerate(sections):
            rank_by_id[section["section_id"]] = rank
            selected_meta.append(
                {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "doc": section.get("doc"),
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "rank": rank + 1,
                    "chunk_count": len(self.store.get_chunks_for_section(section["section_id"])),
                }
            )

        section_chunks = self.store.get_chunks_for_sections([section["section_id"] for section in sections])
        chunks: list[Chunk] = []
        total = max(len(sections), 1)
        for record in section_chunks:
            rank = rank_by_id.get(record.get("section_id"), total)
            score = (total - rank) / total
            chunks.append(
                Chunk(
                    text=record.get("text", ""),
                    chunk_id=record.get("chunk_id", ""),
                    doc=record.get("doc") or "unknown",
                    page=record.get("page_start"),
                    score=score,
                    section_id=record.get("section_id"),
                    section_title=record.get("title"),
                    parent_id=record.get("parent_id"),
                    page_start=record.get("page_start"),
                    page_end=record.get("page_end"),
                    sources=["structural"],
                    metadata={
                        "structural_rank": rank + 1 if rank < total else total,
                        "section_rank_score": score,
                        **((record.get("metadata") or {})),
                    },
                )
            )

        chunks.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        returned_chunks = chunks[:k]
        emit_progress(
            on_event,
            "structural_expand",
            "done" if returned_chunks else "warning",
            (
                f"Expanded {len(sections)} section(s) into {len(returned_chunks)} structural chunk(s)."
                if returned_chunks
                else f"Expanded {len(sections)} section(s), but no structural chunks were available."
            ),
            selected_sections=selected_meta,
            chunk_count=len(returned_chunks),
            matched_chunks=len(chunks),
            strategy=info.get("strategy", "unknown"),
            retries=int(info.get("retries", 0)),
        )
        return StructuralResult(
            chunks=returned_chunks,
            selected_sections=selected_meta,
            strategy=info.get("strategy", "unknown"),
            raw_response=info.get("raw_response", ""),
            retries=int(info.get("retries", 0)),
            reason=info.get("reason", ""),
        )
