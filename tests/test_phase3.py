"""Phase 3 tests: structural selection, validation, and expansion."""

from __future__ import annotations

import pymupdf

from retrieval.structural import StructuralRetriever
from sah_config import SAHConfig
from structure.ingest import ingest_document
from structure.store import StructureStore


class FakeLLM:
    def __init__(self, responses, available=True):
        self.responses = list(responses)
        self.available = available

    def is_available(self):
        return self.available

    def generate(self, prompt, json_mode=False, temperature=0.0):
        return self.responses.pop(0)


def make_store(tmp_path):
    path = tmp_path / "book.pdf"
    pdf = pymupdf.open()
    for text in ["Introduction material", "Quadruples material", "Triples material"]:
        page = pdf.new_page()
        page.insert_text((72, 72), text, fontsize=10)
    pdf.set_toc([[1, "Introduction", 1], [1, "Quadruples", 2], [1, "Triples", 3]])
    pdf.save(path)
    pdf.close()
    store = StructureStore(tmp_path / "test.db", tmp_path / "json")
    ingest_document(path, store)
    doc = store.list_documents()[0]
    return store, doc, store.get_sections(doc["doc_id"])


def test_malformed_json_retries_then_accepts(tmp_path):
    store, doc, sections = make_store(tmp_path)
    valid_id = sections[1]["section_id"]
    llm = FakeLLM(["not json at all", '{"section_ids": ["%s"]}' % valid_id])
    retriever = StructuralRetriever(store, llm=llm, config=SAHConfig(structural_top_n=2))
    result = retriever.retrieve_structural("What are quadruples?", k=5, doc_ids=[doc["doc_id"]])
    assert result.strategy == "llm"
    assert result.retries == 1
    assert result.selected_sections[0]["section_id"] == valid_id
    assert result.chunks and result.chunks[0].section_id == valid_id


def test_invalid_ids_are_dropped(tmp_path):
    store, doc, sections = make_store(tmp_path)
    valid_id = sections[2]["section_id"]
    llm = FakeLLM(['{"section_ids": ["made-up-id", "%s", "%s"]}' % (valid_id, valid_id)])
    retriever = StructuralRetriever(store, llm=llm, config=SAHConfig(structural_top_n=3))
    result = retriever.retrieve_structural("triples", k=5, doc_ids=[doc["doc_id"]])
    assert [item["section_id"] for item in result.selected_sections] == [valid_id]


def test_unavailable_llm_uses_keyword_fallback(tmp_path):
    store, doc, sections = make_store(tmp_path)
    retriever = StructuralRetriever(
        store, llm=FakeLLM([], available=False), config=SAHConfig(structural_top_n=1)
    )
    result = retriever.retrieve_structural("quadruples", k=5, doc_ids=[doc["doc_id"]])
    assert result.strategy == "keyword_fallback"
    assert result.selected_sections[0]["title"] == "Quadruples"


def test_section_rank_scores_are_monotonic(tmp_path):
    store, doc, sections = make_store(tmp_path)
    ids = [section["section_id"] for section in sections]
    llm = FakeLLM(['{"section_ids": ["%s", "%s", "%s"]}' % tuple(ids)])
    retriever = StructuralRetriever(store, llm=llm, config=SAHConfig(structural_top_n=3))
    result = retriever.retrieve_structural("anything", k=5, doc_ids=[doc["doc_id"]])
    scores = [chunk.score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert result.chunks[0].score == 1.0
