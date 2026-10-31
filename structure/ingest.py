"""Bind legacy chunks to extracted sections and persist the structure index."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from structure.chunking import chunks_for_sections
from structure.extractor import extract_document
from structure.models import ExtractedDocument, Section, StructureChunk
from structure.store import StructureStore

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _legacy_records(chunks: list[Any], document: ExtractedDocument) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("page_content") or ""
            metadata = dict(chunk.get("metadata") or {})
            chunk_id_value = chunk.get("chunk_id")
        else:
            text = getattr(chunk, "page_content", "") or ""
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            chunk_id_value = getattr(chunk, "chunk_id", None)
        if not text.strip():
            continue
        chunk_id = chunk_id_value or metadata.get("chunk_id") or f"{document.doc_id}::legacy-{ordinal:04d}"
        page = metadata.get("page_number")
        if page is None and metadata.get("page") is not None:
            try:
                page = int(metadata["page"]) + 1
            except (TypeError, ValueError):
                page = None
        records.append(
            {
                "chunk_id": str(chunk_id),
                "text": text,
                "metadata": metadata,
                "page": int(page) if page is not None else None,
                "ordinal": ordinal,
            }
        )
    return records


def _choose_section(record: dict[str, Any], leaves: list[Section], all_sections: list[Section]) -> Section:
    text = _normalize(record["text"])
    page = record.get("page")

    if page and page > 0:
        candidates = [
            section for section in leaves
            if section.page_start <= page <= section.page_end
        ]
        if candidates:
            # Prefer the narrowest page range; ties go to the first leaf.
            candidates.sort(key=lambda section: (section.page_end - section.page_start, section.ordinal))
            return candidates[0]

    needle = text[:120]
    if needle:
        containing = [
            section for section in leaves
            if needle and needle in _normalize(section.text)
        ]
        if containing:
            containing.sort(key=lambda section: (len(section.text), section.ordinal))
            return containing[0]

    if page and page > 0:
        # A page outside a leaf range can still fall inside a parent range.
        containing = [
            section for section in all_sections
            if section.page_start <= page <= section.page_end
        ]
        if containing:
            containing.sort(key=lambda section: (section.page_end - section.page_start, section.ordinal))
            return containing[0]

    roots = [section for section in all_sections if section.parent_id is None]
    return roots[0] if roots else all_sections[0]


def build_structure_chunks(
    document: ExtractedDocument,
    existing_chunks: list[Any] | None = None,
) -> list[StructureChunk]:
    """Use the legacy chunks when present so IDs stay consistent across indexes."""
    all_sections = document.sections
    parent_ids = {section.parent_id for section in all_sections if section.parent_id}
    leaves = [section for section in all_sections if section.section_id not in parent_ids]
    if not leaves:
        leaves = all_sections

    if existing_chunks:
        records = _legacy_records(existing_chunks, document)
        chunks: list[StructureChunk] = []
        for record in records:
            section = _choose_section(record, leaves, all_sections)
            chunks.append(
                StructureChunk(
                    chunk_id=record["chunk_id"],
                    doc_id=document.doc_id,
                    section_id=section.section_id,
                    text=record["text"],
                    page_start=section.page_start,
                    page_end=section.page_end,
                    ordinal=record["ordinal"],
                    metadata=record["metadata"],
                )
            )
        if chunks:
            return chunks

    return chunks_for_sections(all_sections, document.doc_id)


def ingest_document(
    path: str | Path,
    store: StructureStore,
    existing_chunks: list[Any] | None = None,
) -> dict[str, Any]:
    """Extract, bind, persist, and summarize one document's structure."""
    path = Path(path)
    document = extract_document(path)
    chunks = build_structure_chunks(document, existing_chunks=existing_chunks)
    store.upsert_document(document, chunks)

    mapping: dict[str, list[str]] = {}
    for chunk in chunks:
        mapping.setdefault(chunk.section_id, []).append(chunk.chunk_id)

    logger.info(
        "Stored structure for %s: doc_id=%s sections=%d chunks=%d strategy=%s",
        path.name, document.doc_id, len(document.sections), len(chunks), document.strategy,
    )
    return {
        "doc_id": document.doc_id,
        "strategy": document.strategy,
        "sections": len(document.sections),
        "chunks": len(chunks),
        "section_chunks": mapping,
        "document": document,
    }


def section_tree_text(store: StructureStore, doc_id: str, include_chunks: bool = False) -> str:
    """Render a compact indented tree for CLI inspection."""
    sections = store.get_sections(doc_id)
    if not sections:
        return f"(no sections stored for {doc_id})"
    lines = []
    for section in sections:
        indent = "  " * max(0, int(section.get("level", 1)) - 1)
        line = (
            f"{indent}- [{section['section_id']}] {section['title']} "
            f"(L{section['level']}, pages {section['page_start']}-{section['page_end']})"
        )
        if include_chunks:
            chunk_ids = [chunk["chunk_id"] for chunk in store.get_chunks_for_section(section["section_id"])]
            line += f" chunks={len(chunk_ids)}"
        lines.append(line)
    return "\n".join(lines)
