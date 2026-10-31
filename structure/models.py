"""Typed structure records for SAH-RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """A document section extracted from a ToC, headings, or fallback."""

    section_id: str
    title: str
    level: int
    parent_id: str | None
    page_start: int
    page_end: int
    text: str
    doc_id: str = ""
    ordinal: int = 0
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "level": self.level,
            "parent_id": self.parent_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "ordinal": self.ordinal,
            "strategy": self.strategy,
        }


@dataclass
class ExtractedDocument:
    """Result of structure extraction for one file."""

    doc_id: str
    filename: str
    path: str
    title: str
    strategy: str
    page_count: int
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StructureChunk:
    """A chunk bound to one section."""

    chunk_id: str
    doc_id: str
    section_id: str
    text: str
    page_start: int
    page_end: int
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "ordinal": self.ordinal,
            "metadata": self.metadata,
        }
