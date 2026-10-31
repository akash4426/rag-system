"""Claim-level citation verification with a cross-encoder or NLI model.

The generator is never asked to judge its own answer. Each generated claim is
compared against the evidence it cites, using either the retrieval cross-encoder
or an optional NLI cross-encoder.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

from evidence.reranker import CrossEncoderReranker
from generation.types import Claim, VerificationReport
from sah_config import SAHConfig

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verify claims against their cited evidence passages."""

    def __init__(
        self,
        config: SAHConfig | None = None,
        reranker: CrossEncoderReranker | None = None,
        scorer: Callable[[list[list[str]]], Sequence[float]] | None = None,
    ):
        self.config = config or SAHConfig.from_env()
        self.mode = (
            self.config.citation_verifier
            if self.config.citation_verifier in {"cross-encoder", "nli"}
            else "cross-encoder"
        )
        self.reranker = reranker or CrossEncoderReranker(self.config.reranker_model)
        self._scorer = scorer
        self._nli_model = None
        self._nli_failed = False
        self._entailment_index = 1

    def _threshold(self) -> float:
        if self.mode == "nli":
            return self.config.citation_nli_threshold
        return self.config.citation_threshold

    def _load_nli(self) -> bool:
        if self._nli_model is not None:
            return True
        if self._nli_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder

            self._nli_model = CrossEncoder(self.config.citation_nli_model)
            labels = getattr(self._nli_model.model.config, "id2label", {}) or {}
            for index, label in labels.items():
                if "entail" in str(label).lower():
                    self._entailment_index = int(index)
                    break
            return True
        except Exception as exc:
            logger.warning("NLI verifier unavailable (%s); falling back to cross-encoder.", exc)
            self._nli_failed = True
            return False

    @staticmethod
    def _softmax(row: Sequence[float]) -> list[float]:
        values = [float(value) for value in row]
        peak = max(values) if values else 0.0
        exps = [math.exp(value - peak) for value in values]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def _nli_scores(self, claim: str, texts: list[str]) -> list[float]:
        pairs = [[claim, text] for text in texts]
        predictions = self._nli_model.predict(pairs)
        scores: list[float] = []
        for row in predictions:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, (list, tuple)):
                probabilities = self._softmax(row)
                index = self._entailment_index if self._entailment_index < len(probabilities) else -1
                scores.append(probabilities[index])
            else:
                scores.append(float(row))
        return scores

    def _score(self, claim: str, texts: list[str]) -> tuple[list[float], str]:
        if self._scorer is not None:
            values = self._scorer([[claim, text] for text in texts])
            return [float(value) for value in values], self.mode
        if self.mode == "nli" and self._load_nli():
            try:
                return self._nli_scores(claim, texts), "nli"
            except Exception as exc:
                logger.warning("NLI prediction failed (%s); falling back to cross-encoder.", exc)
        return self.reranker.score(claim, texts), "cross-encoder"

    def verify_claim(self, claim: Claim, evidence_by_id: dict[str, str]) -> Claim:
        """Populate support fields for one claim in place."""
        if not claim.citations:
            claim.supported = False
            claim.score = 0.0
            claim.reason = "No citation marker."
            return claim

        unknown = [evidence_id for evidence_id in claim.citations if evidence_id not in evidence_by_id]
        if unknown:
            claim.supported = False
            claim.score = 0.0
            claim.reason = "Unknown citation IDs: " + ", ".join(unknown)
            return claim

        texts = [evidence_by_id[evidence_id] for evidence_id in claim.citations]
        scores, used_mode = self._score(claim.text, texts)
        claim.score = max(scores) if scores else 0.0
        claim.supported = bool(scores) and claim.score >= self._threshold()
        claim.reason = (
            f"{used_mode} score {claim.score:.2f} "
            f"{'meets' if claim.supported else 'is below'} threshold {self._threshold():.2f}."
        )
        return claim

    def verify(self, claims: list[Claim], evidence_by_id: dict[str, str]) -> VerificationReport:
        """Verify all claims and aggregate citation accuracy."""
        invalid: list[str] = []
        missing = 0
        for claim in claims:
            self.verify_claim(claim, evidence_by_id)
            if not claim.citations:
                missing += 1
            for evidence_id in claim.citations:
                if evidence_id not in evidence_by_id and evidence_id not in invalid:
                    invalid.append(evidence_id)

        total = len(claims)
        supported = sum(1 for claim in claims if claim.supported)
        scored = [claim.score for claim in claims if claim.citations]
        report = VerificationReport(
            claims=claims,
            citation_accuracy=supported / total if total else 0.0,
            mean_score=sum(scored) / len(scored) if scored else 0.0,
            invalid_citations=invalid,
            missing_citations=missing,
            verifier=self.mode,
        )
        logger.info(
            "Citation verification: %d/%d claims supported, invalid=%s",
            supported, total, invalid,
        )
        return report
