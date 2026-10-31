"""Phase 5 tests: expansion, reranking, validation, and retry ablation."""

from __future__ import annotations

from types import SimpleNamespace

from evidence.expander import EvidenceExpander
from evidence.pipeline import EvidencePipeline
from evidence.reranker import CrossEncoderReranker
from evidence.retry import EvidenceRetrier
from evidence.types import ValidationResult
from evidence.validator import EvidenceValidator
from retrieval.controller import ControllerDecision
from retrieval.adaptive import AdaptiveResult
from retrieval.types import Chunk
from sah_config import SAHConfig
from structure.models import ExtractedDocument, Section, StructureChunk
from structure.store import StructureStore


def make_chunk(cid, text, section_id, score=1.0):
    return Chunk(
        text=text, chunk_id=cid, doc="d.pdf", page=1, score=score,
        section_id=section_id, section_title=section_id.upper(), sources=["test"],
    )


def make_store(tmp_path):
    store = StructureStore(tmp_path / "ev.db", tmp_path / "json")
    sections = [
        Section("s1", "Parent", 1, None, 1, 3, "Parent summary text", doc_id="d", ordinal=0),
        Section("s2", "Child A", 2, "s1", 1, 1, "Child A body", doc_id="d", ordinal=1),
        Section("s3", "Child B", 2, "s1", 2, 2, "Child B body", doc_id="d", ordinal=2),
    ]
    chunks = [
        StructureChunk("c2", "d", "s2", "Child A body", 1, 1),
        StructureChunk("c2b", "d", "s2", "More child A body", 1, 1, ordinal=1),
        StructureChunk("c3", "d", "s3", "Child B body", 2, 2),
    ]
    store.upsert_document(
        ExtractedDocument("d", "d.pdf", "d.pdf", "D", "test", 3, sections), chunks
    )
    return store


def good_scorer(pairs):
    return [10.0 if "child" in text.lower() else -6.0 for _, text in pairs]


def test_expander_adds_parent_neighbour_sibling_with_budget(tmp_path):
    store = make_store(tmp_path)
    config = SAHConfig(evidence_expansion=True, context_token_budget=1000)
    expander = EvidenceExpander(store, config)
    primary = [make_chunk("c2b", "More child A body", "s2")]
    evidence = expander.expand("child a", primary, budget=1000)
    roles = [item.metadata["role"] for item in evidence.items]
    assert roles[0] == "primary"
    assert "parent" in roles
    assert "sibling" in roles
    assert evidence.token_count > 0
    assert evidence.items[0].text == "More child A body"
    assert primary[0].text == "More child A body"


def test_expander_respects_tiny_budget(tmp_path):
    store = make_store(tmp_path)
    expander = EvidenceExpander(store, SAHConfig(evidence_expansion=True))
    evidence = expander.expand("child", [make_chunk("c2", "x " * 1000, "s2")], budget=20)
    assert evidence.token_count <= 40
    assert evidence.truncated


def test_reranker_order_and_metadata():
    reranker = CrossEncoderReranker(scorer=good_scorer)
    chunks = [make_chunk("bad", "unrelated text", "s1"), make_chunk("good", "child evidence", "s2")]
    ranked = reranker.rerank("child", chunks, top_k=1)
    assert ranked[0].chunk_id == "good"
    assert ranked[0].metadata["rerank_score"] == 10.0


def test_validator_flags_missing_aspects(tmp_path):
    reranker = CrossEncoderReranker(scorer=good_scorer)
    config = SAHConfig(relevance_threshold=-2.0)
    validator = EvidenceValidator(reranker, config)
    good = validator.validate("child evidence", [make_chunk("c", "child evidence", "s2")])
    assert good.sufficient
    bad = validator.validate("quantum compilers", [make_chunk("c", "child evidence", "s2")])
    assert not bad.sufficient
    assert "quantum" in bad.missing_aspects
    assert validator.validate("anything", []).sufficient is False


def test_structure_guided_retry_uses_neighbourhood(tmp_path):
    store = make_store(tmp_path)
    reranker = CrossEncoderReranker(scorer=good_scorer)
    retrier = EvidenceRetrier(store, reranker, SAHConfig())
    hit = [make_chunk("c2", "Child A body", "s2")]
    new, event = retrier.structure_guided("child", ["sibling"], hit, k=5, attempt=1)
    assert event["mode"] == "structure"
    assert "s1" in event["neighbourhood"]
    assert new and all(chunk.chunk_id != "c2" for chunk in new)


def test_plain_retry_requeries_adaptive(tmp_path):
    store = make_store(tmp_path)
    retrier = EvidenceRetrier(store, CrossEncoderReranker(scorer=good_scorer), SAHConfig())
    calls = []

    class FakeAdaptive:
        def retrieve_adaptive(self, query, k=5, doc_ids=None):
            calls.append(query)
            return SimpleNamespace(chunks=[make_chunk("new", "quantum evidence", "s3")])

    new, event = retrier.plain("original", ["quantum"], FakeAdaptive(), k=3)
    assert event["mode"] == "plain"
    assert "quantum" in calls[0]
    assert new[0].chunk_id == "new"


def test_pipeline_structure_retry_event(tmp_path):
    store = make_store(tmp_path)

    class FakeAdaptive:
        def __init__(self):
            self.queries = []

        def retrieve_adaptive(self, query, k=5, doc_ids=None, fusion_pool=None):
            self.queries.append(query)
            decision = ControllerDecision(["semantic", "lexical"], k, "simple", "test")
            return AdaptiveResult(
                chunks=[make_chunk("c2", "Child A body", "s2")],
                decision=decision,
                counts={"semantic": 1, "lexical": 0},
            )

    class FakeValidator:
        def __init__(self):
            self.calls = 0

        def validate(self, query, chunks, context_text=""):
            self.calls += 1
            if self.calls == 1:
                return ValidationResult(False, ["sibling"], ["missing"], max_score=-9.0)
            return ValidationResult(True, [], ["ok"], max_score=5.0)

    config = SAHConfig(
        adaptive_controller=True, structural_retrieval=True, rerank=True,
        evidence_expansion=True, evidence_validation=True, structure_guided_retry=True,
        max_retries=2, top_k=3,
    )
    adaptive = FakeAdaptive()
    pipeline = EvidencePipeline(
        adaptive, store, config,
        reranker=CrossEncoderReranker(scorer=good_scorer),
        validator=FakeValidator(),
    )
    result = pipeline.gather("child a", k=3)
    assert result.validation.sufficient
    assert result.retries and result.retries[0]["mode"] == "structure"
    assert result.retries[0]["sufficient_after"] is True
    assert result.chunks


def test_pipeline_plain_retry_ablation(tmp_path):
    store = make_store(tmp_path)

    class FakeAdaptive:
        def __init__(self):
            self.queries = []

        def retrieve_adaptive(self, query, k=5, doc_ids=None, fusion_pool=None):
            self.queries.append(query)
            decision = ControllerDecision(["semantic", "lexical"], k, "simple", "test")
            chunks = [make_chunk("c2", "Child A body", "s2")]
            if "sibling" in query:
                chunks = [make_chunk("c3", "Child B body", "s3")]
            return AdaptiveResult(chunks=chunks, decision=decision, counts={"semantic": 1})

    class FakeValidator:
        def __init__(self):
            self.calls = 0

        def validate(self, query, chunks, context_text=""):
            self.calls += 1
            if self.calls == 1:
                return ValidationResult(False, ["sibling"], ["missing"], max_score=-9.0)
            return ValidationResult(True, [], ["ok"], max_score=5.0)

    config = SAHConfig(
        adaptive_controller=True, structural_retrieval=True, rerank=True,
        evidence_expansion=False, evidence_validation=True, structure_guided_retry=False,
        max_retries=2, top_k=3,
    )
    adaptive = FakeAdaptive()
    pipeline = EvidencePipeline(
        adaptive, store, config,
        reranker=CrossEncoderReranker(scorer=good_scorer),
        validator=FakeValidator(),
    )
    result = pipeline.gather("child", k=3)
    assert result.retries[0]["mode"] == "plain"
    assert any("sibling" in query for query in adaptive.queries)
