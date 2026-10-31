"""Phase 4 tests: controller routing, RRF fusion, and parallel adaptive runs."""

from __future__ import annotations

from types import SimpleNamespace

from retrieval.adaptive import AdaptiveRetriever
from retrieval.controller import QueryController
from retrieval.fusion import rrf_fuse
from retrieval.types import Chunk
from sah_config import SAHConfig


def chunk(cid, text, score=1.0, source="x"):
    return Chunk(text=text, chunk_id=cid, doc="d.pdf", page=1, score=score, sources=[source])


class FakeBaseline:
    def retrieve_dense(self, query, k=5):
        return [chunk("c1", "alpha beta", 0.9, "vector_db"), chunk("c2", "gamma", 0.8, "vector_db")]

    def retrieve_sparse(self, query, k=5):
        return [chunk("c1", "alpha beta", 2.0, "bm25"), chunk("c3", "delta", 1.0, "bm25")]


class FakeStructural:
    def __init__(self, fail=False):
        self.fail = fail

    def retrieve_structural(self, query, k=5, doc_ids=None):
        if self.fail:
            raise RuntimeError("structural unavailable")
        return SimpleNamespace(
            chunks=[chunk("c4", "quadruples section", 1.0, "structural")],
            selected_sections=[{"section_id": "s4", "title": "Quadruples", "rank": 1}],
            strategy="llm",
            reason="selected",
            retries=0,
        )


def test_controller_signals():
    controller = QueryController()
    simple = controller.analyze("what is three address code?")
    assert simple.retrievers == ["semantic", "lexical"]
    assert simple.complexity == "simple"

    compare = controller.analyze("compare quadruples and triples and explain the differences")
    assert "structural" in compare.retrievers
    assert compare.k >= 8
    assert compare.complexity == "complex"

    section = controller.analyze("which section discusses indirect triples?")
    assert section.retrievers[0] == "structural"
    assert section.complexity == "moderate"
    assert section.signals["structure_terms"]


def test_rrf_deduplicates_and_merges_sources():
    fused = rrf_fuse({"semantic": [chunk("c1", "alpha", 0.9), chunk("c2", "beta", 0.8)],
                      "lexical": [chunk("c1", "alpha", 2.0), chunk("c3", "gamma", 1.0)]}, top_k=5)
    assert fused[0].chunk_id == "c1"
    assert set(fused[0].sources) == {"semantic", "lexical"}
    assert len({item.chunk_id for item in fused}) == 3
    assert "rrf_score" in fused[0].metadata


def test_adaptive_runs_selected_retrievers():
    config = SAHConfig(adaptive_controller=True, structural_retrieval=True, top_k=5)
    retriever = AdaptiveRetriever(FakeBaseline(), FakeStructural(), config=config)
    result = retriever.retrieve_adaptive(
        "which section compares quadruples and triples and explains the differences in representation?"
    )
    assert set(result.decision.retrievers) == {"structural", "semantic", "lexical"}
    assert result.counts == {"semantic": 2, "lexical": 2, "structural": 1}
    assert result.selected_sections[0]["title"] == "Quadruples"
    assert result.chunks and result.chunks[0].chunk_id == "c1"
    assert result.structural_strategy == "llm"


def test_structural_failure_does_not_break_fusion():
    config = SAHConfig(adaptive_controller=True, structural_retrieval=True, top_k=5)
    retriever = AdaptiveRetriever(FakeBaseline(), FakeStructural(fail=True), config=config)
    result = retriever.retrieve_adaptive("which section compares quadruples and triples?")
    assert "structural" in result.errors
    assert result.chunks
    assert not result.selected_sections


def test_fixed_mode_omits_structural_when_flag_off():
    config = SAHConfig(adaptive_controller=False, structural_retrieval=False, top_k=5)
    retriever = AdaptiveRetriever(FakeBaseline(), FakeStructural(), config=config)
    result = retriever.retrieve_adaptive("compare alpha and beta")
    assert result.decision.retrievers == ["semantic", "lexical"]
