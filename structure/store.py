"""SQLite + JSON persistence for document structure and section-aware chunks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from structure.models import ExtractedDocument, Section, StructureChunk


def _text_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("metadata_json",):
        if key in data:
            try:
                data["metadata"] = json.loads(data.pop(key) or "{}")
            except json.JSONDecodeError:
                data["metadata"] = {}
    return data


class StructureStore:
    """Persist the section tree and chunk mapping for every document."""

    def __init__(self, db_path: str | Path = "data/sah_rag.db", json_dir: str | Path = "data/structure"):
        self.db_path = str(db_path)
        self.json_dir = Path(json_dir)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT,
                    title TEXT,
                    strategy TEXT,
                    page_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS sections (
                    section_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT,
                    level INTEGER,
                    parent_id TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    text TEXT,
                    ordinal INTEGER,
                    strategy TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    section_id TEXT,
                    text TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    ordinal INTEGER,
                    text_hash TEXT,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_sections_parent ON sections(parent_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(text_hash);
                """
            )

    # ------------------------------------------------------------- persistence
    def upsert_document(self, document: ExtractedDocument, chunks: list[StructureChunk]) -> None:
        """Replace a document's tree/chunks and write its JSON mirror."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (document.doc_id,))
            conn.execute("DELETE FROM sections WHERE doc_id = ?", (document.doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (document.doc_id,))
            conn.execute(
                """INSERT INTO documents
                   (doc_id, filename, path, title, strategy, page_count, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.doc_id, document.filename, document.path, document.title,
                    document.strategy, document.page_count, now,
                    json.dumps({"warnings": document.warnings}),
                ),
            )
            conn.executemany(
                """INSERT INTO sections
                   (section_id, doc_id, title, level, parent_id, page_start, page_end, text, ordinal, strategy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        section.section_id, section.doc_id, section.title, section.level,
                        section.parent_id, section.page_start, section.page_end, section.text,
                        section.ordinal, section.strategy,
                    )
                    for section in document.sections
                ],
            )
            conn.executemany(
                """INSERT INTO chunks
                   (chunk_id, doc_id, section_id, text, page_start, page_end, ordinal, text_hash, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.chunk_id, chunk.doc_id, chunk.section_id, chunk.text,
                        chunk.page_start, chunk.page_end, chunk.ordinal, _text_hash(chunk.text),
                        json.dumps(chunk.metadata or {}),
                    )
                    for chunk in chunks
                ],
            )

        payload = {
            "document": {
                "doc_id": document.doc_id,
                "filename": document.filename,
                "path": document.path,
                "title": document.title,
                "strategy": document.strategy,
                "page_count": document.page_count,
                "created_at": now,
                "warnings": document.warnings,
            },
            "sections": [section.to_dict() for section in document.sections],
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        json_path = self.json_dir / f"{document.doc_id}.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- documents
    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT d.*,
                          (SELECT COUNT(*) FROM sections s WHERE s.doc_id = d.doc_id) AS section_count,
                          (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.doc_id) AS chunk_count
                   FROM documents d ORDER BY d.filename"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def get_document_by_filename(self, filename: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename = ? ORDER BY created_at DESC LIMIT 1",
                (filename,),
            ).fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------- sections
    def get_sections(self, doc_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sections WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_section(self, section_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.*, d.filename AS doc
                   FROM sections s LEFT JOIN documents d ON d.doc_id = s.doc_id
                   WHERE s.section_id = ?""",
                (section_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_leaf_sections(self, doc_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM sections
                   WHERE doc_id = ?
                     AND section_id NOT IN (SELECT parent_id FROM sections WHERE doc_id = ? AND parent_id IS NOT NULL)
                   ORDER BY ordinal""",
                (doc_id, doc_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_children(self, section_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sections WHERE parent_id = ? ORDER BY ordinal", (section_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_parent(self, section_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT parent.* FROM sections child
                   JOIN sections parent ON parent.section_id = child.parent_id
                   WHERE child.section_id = ?""",
                (section_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_siblings(self, section_id: str) -> list[dict[str, Any]]:
        section = self.get_section(section_id)
        if not section:
            return []
        with self._connect() as conn:
            if section.get("parent_id"):
                rows = conn.execute(
                    """SELECT * FROM sections
                       WHERE parent_id = ? AND section_id != ? ORDER BY ordinal""",
                    (section["parent_id"], section_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM sections
                       WHERE doc_id = ? AND parent_id IS NULL AND section_id != ?
                       ORDER BY ordinal""",
                    (section["doc_id"], section_id),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_neighbourhood(self, section_id: str, adjacent: int = 1) -> list[dict[str, Any]]:
        """Parent, siblings, and adjacent sections used by structure-guided retry."""
        section = self.get_section(section_id)
        if not section:
            return []
        found: dict[str, dict[str, Any]] = {section_id: section}
        parent = self.get_parent(section_id)
        if parent:
            found[parent["section_id"]] = parent
        for sibling in self.get_siblings(section_id):
            found[sibling["section_id"]] = sibling

        all_sections = self.get_sections(section["doc_id"])
        index = next((i for i, item in enumerate(all_sections) if item["section_id"] == section_id), None)
        if index is not None:
            for offset in range(1, adjacent + 1):
                for neighbour_index in (index - offset, index + offset):
                    if 0 <= neighbour_index < len(all_sections):
                        item = all_sections[neighbour_index]
                        found[item["section_id"]] = item
        return list(found.values())

    # ------------------------------------------------------------------ chunks
    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT c.*, s.title, s.parent_id, s.level,
                          d.filename AS doc
                   FROM chunks c
                   LEFT JOIN sections s ON s.section_id = c.section_id
                   LEFT JOIN documents d ON d.doc_id = c.doc_id
                   WHERE c.chunk_id = ?""",
                (chunk_id,),
            ).fetchone()
        return _row_dict(row)

    def find_chunk_by_text(self, text: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT c.*, s.title, s.parent_id, s.level,
                          d.filename AS doc
                   FROM chunks c
                   LEFT JOIN sections s ON s.section_id = c.section_id
                   LEFT JOIN documents d ON d.doc_id = c.doc_id
                   WHERE c.text_hash = ? LIMIT 1""",
                (_text_hash(text),),
            ).fetchone()
        return _row_dict(row)

    def get_chunks_for_section(self, section_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.*, s.title, s.parent_id, s.level,
                          d.filename AS doc
                   FROM chunks c
                   LEFT JOIN sections s ON s.section_id = c.section_id
                   LEFT JOIN documents d ON d.doc_id = c.doc_id
                   WHERE c.section_id = ? ORDER BY c.ordinal""",
                (section_id,),
            ).fetchall()
        return [_row_dict(row) for row in rows]  # type: ignore[list-item]

    def get_chunks_for_sections(self, section_ids: list[str]) -> list[dict[str, Any]]:
        if not section_ids:
            return []
        placeholders = ",".join("?" for _ in section_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT c.*, s.title, s.parent_id, s.level,
                           d.filename AS doc
                    FROM chunks c
                    LEFT JOIN sections s ON s.section_id = c.section_id
                    LEFT JOIN documents d ON d.doc_id = c.doc_id
                    WHERE c.section_id IN ({placeholders})
                    ORDER BY s.ordinal, c.ordinal""",
                tuple(section_ids),
            ).fetchall()
        return [_row_dict(row) for row in rows]  # type: ignore[list-item]

    def get_neighbour_chunks(self, chunk_id: str, radius: int = 1) -> list[dict[str, Any]]:
        """Adjacent chunks in the same section plus the section's parent summary."""
        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return []
        section_id = chunk.get("section_id")
        ordinal = int(chunk.get("ordinal") or 0)
        results: list[dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.*, s.title, s.parent_id, s.level,
                          d.filename AS doc
                   FROM chunks c
                   LEFT JOIN sections s ON s.section_id = c.section_id
                   LEFT JOIN documents d ON d.doc_id = c.doc_id
                   WHERE c.section_id = ? AND c.ordinal BETWEEN ? AND ?
                   ORDER BY c.ordinal""",
                (section_id, ordinal - radius, ordinal + radius),
            ).fetchall()
        results.extend(_row_dict(row) for row in rows)  # type: ignore[arg-type]
        parent = self.get_parent(section_id) if section_id else None
        if parent:
            results.append(
                {
                    "chunk_id": f"{parent['section_id']}::summary",
                    "doc_id": parent["doc_id"],
                    "section_id": parent["section_id"],
                    "text": parent.get("text", ""),
                    "title": parent["title"],
                    "parent_id": parent.get("parent_id"),
                    "level": parent.get("level"),
                    "page_start": parent.get("page_start"),
                    "page_end": parent.get("page_end"),
                    "ordinal": -1,
                    "is_context": True,
                    "metadata": {},
                }
            )
        return results

    def find_section_by_title(self, doc_id: str, title: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM sections
                   WHERE doc_id = ? AND lower(trim(title)) = lower(trim(?))
                   ORDER BY ordinal LIMIT 1""",
                (doc_id, title),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            sections = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": int(documents), "sections": int(sections), "chunks": int(chunks)}
