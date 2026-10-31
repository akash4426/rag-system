"""Phase 7 tests: trace and citation serialization for the API/UI."""

from __future__ import annotations

from api.trace import (
    citation_from_chunk,
    citation_from_legacy,
    evidence_context_chunks,
    legacy_context_chunks,
    legacy_trace,
)
from evidence.types import EvidenceSet
from retrieval.types import Chunk


def make_chunk(cid="c1", evidence_id="E1"):
    return Chunk(
        text="Three-address code evidence.",
        chunk_id=cid,
        doc="doc.pdf",
        page=11,
        score=2.0,
        section_id="s1",
        section_title="Slide 11: Three-Address Code",
        metadata={"evidence_id": evidence_id, "role": "primary"},
    )


def test_citation_from_chunk_carries_provenance_and_text():
    citation = citation_from_chunk(make_chunk(), 1)
    assert citation["evidence_id"] == "E1"
    assert citation["chunk_id"] == "c1"
    assert citation["section_title"].endswith("Three-Address Code")
    assert citation["page"] == 11
    assert citation["text"].startswith("Three-address")


def test_legacy_citation_and_context_shape():
    item = {
        "content": "legacy text",
        "metadata": {"source": "data/notes.pdf", "page": 2},
        "sources": ["bm25", "vector_db"],
        "rerank_score": 1.5,
    }
    citation = citation_from_legacy(item, 2)
    assert citation["evidence_id"] == "S2"
    assert citation["doc"] == "notes.pdf"
    assert citation["page"] == 3
    chunks = legacy_context_chunks([item])
    assert chunks[0]["source"].startswith("notes.pdf")
    assert chunks[0]["text"] == "legacy text"


def test_evidence_context_chunks_include_all_roles():
    primary = make_chunk()
    parent = make_chunk("p1", "E2")
    parent.metadata["role"] = "parent"
    evidence = EvidenceSet(items=[primary, parent], primary=[primary], context_text="x")
    chunks = evidence_context_chunks(evidence)
    assert [chunk["role"] for chunk in chunks] == ["primary", "parent"]
    assert chunks[1]["evidence_id"] == "E2"


def test_legacy_trace_has_retrieval_and_generation_sections():
    item = {"content": "text", "metadata": {}, "sources": ["bm25"]}
    trace = legacy_trace("factual", ["text"], [item], 0.6, 0.4, candidate_count=9)
    assert trace["mode"] == "baseline"
    assert trace["retrieval"]["candidates"] == 9
    assert trace["retrieval"]["weights"] == {"sparse": 0.6, "dense": 0.4}
    assert trace["generation"]["grounded"] is False
    assert trace["retrieval"]["chunks"][0]["evidence_id"] == "S1"
