"""Phase 2 tests: PDF/DOCX/Markdown extraction, storage, and chunk mapping."""

from __future__ import annotations

import json

import pymupdf
from docx import Document as DocxDocument

from structure.extractor import extract_document, extract_structure
from structure.ingest import ingest_document, section_tree_text
from structure.store import StructureStore


def make_pdf(path, pages_text, toc=None):
    pdf = pymupdf.open()
    for text in pages_text:
        page = pdf.new_page()
        page.insert_text((72, 72), text, fontsize=10)
    if toc:
        pdf.set_toc(toc)
    pdf.save(path)
    pdf.close()


def test_pdf_bookmarks_store_and_tree(tmp_path):
    path = tmp_path / "sample.pdf"
    make_pdf(
        path,
        ["Intro page text", "Details page text", "Sub details page text"],
        toc=[[1, "Introduction", 1], [1, "Details", 2], [2, "Sub Details", 2]],
    )
    result = extract_document(path)
    assert result.strategy == "pdf_bookmarks"
    assert [section.title for section in result.sections] == ["Introduction", "Details", "Sub Details"]
    assert result.sections[2].parent_id == result.sections[1].section_id
    assert extract_structure(path)[0].section_id.startswith(result.doc_id)

    store = StructureStore(tmp_path / "test.db", tmp_path / "json")
    info = ingest_document(path, store)
    assert info["sections"] == 3
    assert info["chunks"] >= 1
    chunks = store.get_chunks_for_section(result.sections[2].section_id)
    assert chunks and chunks[0]["section_id"] == result.sections[2].section_id
    assert "Introduction" in section_tree_text(store, result.doc_id)
    assert (tmp_path / "json" / f"{result.doc_id}.json").exists()
    payload = json.loads((tmp_path / "json" / f"{result.doc_id}.json").read_text())
    assert payload["document"]["strategy"] == "pdf_bookmarks"


def test_legacy_chunks_keep_ids_and_page_mapping(tmp_path):
    path = tmp_path / "mapped.pdf"
    make_pdf(path, ["alpha content", "beta content", "gamma content"], toc=[[1, "Alpha", 1], [1, "Beta", 2]])
    store = StructureStore(tmp_path / "mapped.db", tmp_path / "json")
    legacy = [
        {"chunk_id": "legacy-0", "page_content": "alpha content", "metadata": {"page": 0}},
        {"chunk_id": "legacy-1", "page_content": "beta content", "metadata": {"page": 1}},
        {"chunk_id": "legacy-2", "page_content": "gamma content", "metadata": {"page": 2}},
    ]
    info = ingest_document(path, store, existing_chunks=legacy)
    assert info["chunks"] == 3
    first = store.get_chunk("legacy-0")
    assert first["section_id"] == info["document"].sections[0].section_id
    third = store.get_chunk("legacy-2")
    assert third["section_id"] == info["document"].sections[1].section_id
    assert store.find_chunk_by_text("beta content")["chunk_id"] == "legacy-1"


def test_docx_heading_hierarchy(tmp_path):
    path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_heading("Chapter One", level=1)
    doc.add_paragraph("Chapter body.")
    doc.add_heading("Section One", level=2)
    doc.add_paragraph("Section body.")
    doc.save(path)

    result = extract_document(path)
    assert result.strategy == "docx_headings"
    assert [section.level for section in result.sections] == [1, 2]
    assert result.sections[1].parent_id == result.sections[0].section_id

    store = StructureStore(tmp_path / "docx.db", tmp_path / "json")
    info = ingest_document(path, store)
    assert info["chunks"] >= 1
    assert store.get_chunks_for_section(result.sections[1].section_id)


def test_markdown_and_text_headings(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Top\nintro\n## Child\nbody\n", encoding="utf-8")
    result = extract_document(md)
    assert result.strategy == "markdown_headings"
    assert result.sections[1].parent_id == result.sections[0].section_id

    txt = tmp_path / "plain.txt"
    txt.write_text("Chapter 1 Introduction\nhello\n", encoding="utf-8")
    txt_result = extract_document(txt)
    assert txt_result.strategy == "text_headings"


def test_pdf_falls_back_to_root_without_headings(tmp_path):
    path = tmp_path / "plain.pdf"
    make_pdf(path, ["just ordinary body text"])
    result = extract_document(path)
    assert result.strategy in {"fixed_chunks", "pdf_headings"}
    store = StructureStore(tmp_path / "fallback.db", tmp_path / "json")
    info = ingest_document(path, store)
    assert info["sections"] >= 1
    assert info["chunks"] >= 1
