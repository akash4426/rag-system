"""Phase 6 tests: grounded generation, citations, verification, abstention."""

from __future__ import annotations

from evidence.types import EvidenceSet, ValidationResult
from generation.answerer import ABSTENTION_MESSAGE, GroundedAnswerer
from generation.claims import extract_citation_ids, split_claims
from generation.generator import GroundedGenerator
from generation.prompt import INSUFFICIENT_SENTINEL, GroundedPromptBuilder
from generation.verifier import CitationVerifier
from retrieval.types import Chunk
from sah_config import SAHConfig, config_for_experiment


def make_chunk(cid, text, section_id, evidence_id, score=1.0):
    return Chunk(
        text=text, chunk_id=cid, doc="d.pdf", page=1, score=score,
        section_id=section_id, section_title=section_id.upper(), sources=["test"],
        metadata={"evidence_id": evidence_id, "role": "primary"},
    )


def make_evidence():
    items = [
        make_chunk("c1", "Three-address code uses at most one operator and three operands.", "s1", "E1"),
        make_chunk("c2", "Quadruples store an operation, two operands, and a result.", "s2", "E2"),
    ]
    context = "\n\n".join(f"[{item.metadata['evidence_id']}] {item.text}" for item in items)
    return EvidenceSet(items=items, primary=items, context_text=context, token_count=40)


def make_scorer(pairs):
    return [
        8.0 if "three-address" in claim.lower() and "three-address" in text.lower() else -5.0
        for claim, text in pairs
    ]


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, json_mode=False, temperature=0.0):
        self.prompts.append(prompt)
        return self.response


def make_answerer(response, config=None, scorer=make_scorer):
    config = config or SAHConfig(
        grounded_generation=True, citation_verification=True, citation_threshold=0.0
    )
    client = FakeClient(response)
    generator = GroundedGenerator(config, client=client)
    verifier = CitationVerifier(config, scorer=scorer)
    return GroundedAnswerer(config, generator=generator, verifier=verifier), client


def test_split_claims_and_extract_citations():
    text = "Three-address code has one operator [E1]. It has three operands [E1][E2].\n- A bullet claim [E2]."
    claims = split_claims(text)
    assert len(claims) == 3
    assert extract_citation_ids(claims[1]) == ["E1", "E2"]
    assert extract_citation_ids(claims[2]) == ["E2"]
    assert extract_citation_ids("no citations here") == []


def test_prompt_contains_rules_and_evidence_ids():
    prompt = GroundedPromptBuilder().build("What is three-address code?", make_evidence())
    assert "[E1]" in prompt and "[E2]" in prompt
    assert "Use only the evidence" in prompt
    assert INSUFFICIENT_SENTINEL in prompt


def test_answerer_keeps_supported_claims_and_resolves_citations():
    answerer, client = make_answerer(
        "Three-address code uses one operator [E1]. Quantum compilers need qubits [E2]."
    )
    result = answerer.answer("What is three-address code?", make_evidence())
    assert not result.abstained
    assert "Three-address code" in result.answer
    assert "Quantum" not in result.answer
    assert [citation.chunk_id for citation in result.citations] == ["c1"]
    assert result.verification.claims_total == 2
    assert result.verification.claims_supported == 1
    assert client.prompts


def test_answerer_abstains_when_no_claim_is_supported():
    answerer, _ = make_answerer("Quantum compilers need qubits [E1].")
    result = answerer.answer("quantum compilers", make_evidence())
    assert result.abstained
    assert result.answer.startswith(ABSTENTION_MESSAGE)
    assert result.verification.claims_supported == 0


def test_answerer_skips_generation_when_validation_insufficient():
    answerer, client = make_answerer("unused [E1].")
    validation = ValidationResult(
        sufficient=False,
        missing_aspects=["quantum"],
        reasons=["missing"],
        aspects=["advantages", "quantum"],
    )
    result = answerer.answer("quantum compilers", make_evidence(), validation)
    assert result.abstained and not result.generated
    assert client.prompts == []
    assert result.supported_parts == ["advantages"]
    assert "does not provide evidence for: quantum" in result.answer


def test_answerer_honours_insufficient_sentinel():
    answerer, _ = make_answerer(f"{INSUFFICIENT_SENTINEL} The document has no quantum content.")
    result = answerer.answer("quantum compilers", make_evidence())
    assert result.abstained
    assert "no quantum content" in result.abstention_reason


def test_answerer_reports_invalid_citation_and_missing_citation():
    answerer, _ = make_answerer("Three-address code is explained here [E9]. It also has operands.")
    result = answerer.answer("three-address code", make_evidence())
    assert result.abstained
    assert result.verification.invalid_citations == ["E9"]
    assert result.verification.missing_citations == 1


def test_verification_flag_is_switchable():
    config = SAHConfig(grounded_generation=True, citation_verification=False)
    answerer, _ = make_answerer("Quantum compilers need qubits [E1].", config=config)
    result = answerer.answer("quantum", make_evidence())
    assert not result.abstained
    assert result.verification is None
    assert result.claims[0].supported


def test_e5_experiment_enables_generation_and_verification():
    config = config_for_experiment("E5", SAHConfig())
    assert config.grounded_generation and config.citation_verification
    assert config.evidence_validation and config.structure_guided_retry
