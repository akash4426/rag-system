"""Small recursive text chunker used when no legacy chunks are available."""

from __future__ import annotations

import hashlib

from structure.models import Section, StructureChunk


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split on paragraph/sentence boundaries with a hard character fallback."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    units = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in paragraph.replace("! ", "!\n").replace("? ", "?\n").replace(". ", ".\n").splitlines():
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            chunks.append(unit[:chunk_size])
            current = unit[chunk_size - overlap:] if overlap > 0 else ""
    if current:
        chunks.append(current)
    return chunks


def chunks_for_sections(
    sections: list[Section],
    doc_id: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[StructureChunk]:
    """Create chunks for leaf sections (parents are context summaries only)."""
    parent_ids = {section.parent_id for section in sections if section.parent_id}
    leaves = [section for section in sections if section.section_id not in parent_ids]
    if not leaves:
        leaves = sections

    chunks: list[StructureChunk] = []
    counter = 0
    for section in leaves:
        for piece in split_text(section.text, chunk_size=chunk_size, overlap=overlap):
            digest = hashlib.sha1(f"{doc_id}:{section.section_id}:{counter}".encode()).hexdigest()[:10]
            chunks.append(
                StructureChunk(
                    chunk_id=f"{section.section_id}::c{counter:03d}-{digest}",
                    doc_id=doc_id,
                    section_id=section.section_id,
                    text=piece,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    ordinal=counter,
                )
            )
            counter += 1
    return chunks
