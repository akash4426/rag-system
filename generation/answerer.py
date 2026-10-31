"""Grounding orchestration: generate, verify citations, and abstain safely."""

from __future__ import annotations

import logging

from evidence.types import EvidenceSet, ValidationResult
from generation.claims import split_claim_records
from generation.generator import GroundedGenerator
from generation.prompt import INSUFFICIENT_SENTINEL
from generation.types import Citation, Claim, GroundedAnswer
from generation.verifier import CitationVerifier
from sah_config import SAHConfig

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = "I could not find enough evidence in the indexed documents to answer this question."


class GroundedAnswerer:
    """Generate cited answers and drop claims that evidence does not support."""

    def __init__(
        self,
        config: SAHConfig | None = None,
        generator: GroundedGenerator | None = None,
        verifier: CitationVerifier | None = None,
    ):
        self.config = config or SAHConfig.from_env()
        self.generator = generator or GroundedGenerator(self.config)
        self.verifier = verifier or CitationVerifier(self.config)

    @staticmethod
    def _evidence_map(evidence: EvidenceSet) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for item in evidence.items:
            evidence_id = item.metadata.get("evidence_id")
            if evidence_id:
                mapping[str(evidence_id)] = item.text
        return mapping

    @staticmethod
    def _supported_parts(validation: ValidationResult | None) -> list[str]:
        if not validation or not validation.aspects:
            return []
        missing = set(validation.missing_aspects)
        return [aspect for aspect in validation.aspects if aspect not in missing][:8]

    def _abstain(
        self,
        query: str,
        reason: str,
        validation: ValidationResult | None,
        generated: bool,
        raw_response: str = "",
        claims: list[Claim] | None = None,
        verification=None,
    ) -> GroundedAnswer:
        supported_parts = self._supported_parts(validation)
        lines = [ABSTENTION_MESSAGE]
        if supported_parts:
            lines.append("Retrieved context touches on: " + ", ".join(supported_parts) + ".")
        if validation and validation.missing_aspects:
            lines.append("It does not provide evidence for: " + ", ".join(validation.missing_aspects) + ".")
        logger.info("Grounded answer abstained: %s", reason)
        return GroundedAnswer(
            query=query,
            answer=" ".join(lines),
            abstained=True,
            abstention_reason=reason,
            claims=claims or [],
            supported_parts=supported_parts,
            raw_response=raw_response,
            generated=generated,
            verification=verification,
        )

    @staticmethod
    def _sentinel_reason(raw: str) -> str:
        upper = raw.upper()
        index = upper.find(INSUFFICIENT_SENTINEL)
        if index == -1:
            return ""
        tail = raw[index + len(INSUFFICIENT_SENTINEL):].strip(" \t\r\n.:-")
        return tail or "The generation model reported insufficient evidence."

    def answer(
        self,
        query: str,
        evidence: EvidenceSet,
        validation: ValidationResult | None = None,
    ) -> GroundedAnswer:
        """Answer from evidence, verifying every retained claim."""
        if self.config.abstain_on_insufficient and validation and not validation.sufficient:
            return self._abstain(
                query,
                "Evidence validation marked the evidence insufficient.",
                validation,
                generated=False,
            )

        if not evidence.items:
            return self._abstain(query, "No evidence passages were available.", validation, generated=False)

        try:
            raw = self.generator.generate(query, evidence)
        except Exception as exc:  # generation must fail closed
            logger.warning("Grounded generation failed: %s", exc)
            return self._abstain(
                query,
                f"Generation model unavailable: {exc}",
                validation,
                generated=False,
            )

        sentinel_reason = self._sentinel_reason(raw)
        if sentinel_reason:
            return self._abstain(
                query, sentinel_reason, validation, generated=True, raw_response=raw
            )

        claims = [
            Claim(text=text, citations=citations)
            for text, citations in split_claim_records(raw)
        ]
        if not claims:
            return self._abstain(
                query,
                "The generation model returned no verifiable claims.",
                validation,
                generated=True,
                raw_response=raw,
            )

        verification = None
        retained = claims
        if self.config.citation_verification:
            evidence_by_id = self._evidence_map(evidence)
            verification = self.verifier.verify(claims, evidence_by_id)
            retained = [claim for claim in claims if claim.supported]

        if not retained:
            return self._abstain(
                query,
                "No generated claim could be verified against its cited evidence.",
                validation,
                generated=True,
                raw_response=raw,
                claims=claims,
                verification=verification,
            )

        cited_ids: list[str] = []
        for claim in retained:
            for evidence_id in claim.citations:
                if evidence_id not in cited_ids:
                    cited_ids.append(evidence_id)

        evidence_by_id_chunks = {
            str(item.metadata.get("evidence_id")): item
            for item in evidence.items
            if item.metadata.get("evidence_id")
        }
        citations = [
            Citation.from_chunk(evidence_by_id_chunks[evidence_id])
            for evidence_id in cited_ids
            if evidence_id in evidence_by_id_chunks
        ]

        answer = " ".join(claim.text.strip() for claim in retained).strip()
        return GroundedAnswer(
            query=query,
            answer=answer,
            abstained=False,
            claims=claims,
            citations=citations,
            verification=verification,
            raw_response=raw,
            generated=True,
        )
