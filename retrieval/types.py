"""Shared typed retrieval records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrieved passage with stable provenance.

    ``score`` is always the score of the retriever that produced this record.
    Fused records use ``fused_score`` in ``metadata`` and keep ``score`` as the
    best original score.
    """

    text: str
    chunk_id: str
    doc: str
    page: int | None = None
    score: float = 0.0
    section_id: str | None = None
    section_title: str | None = None
    parent_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, text_limit: int | None = None) -> dict[str, Any]:
        """Serialize for API/trace output, optionally truncating text."""
        data = asdict(self)
        if text_limit is not None and len(data["text"]) > text_limit:
            data["text"] = data["text"][:text_limit] + "..."
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Build a Chunk from a serialized mapping."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
