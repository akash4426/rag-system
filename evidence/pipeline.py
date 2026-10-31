"""Evidence pipeline: retrieve, rerank, expand, validate, and retry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from evidence.expander import EvidenceExpander
from evidence.reranker import CrossEncoderReranker
from evidence.retry import EvidenceRetrier
from evidence.types import EvidenceSet, ValidationResult
from evidence.validator import EvidenceValidator
from retrieval.adaptive import AdaptiveResult, AdaptiveRetriever
from retrieval.types import Chunk
from sah_config import SAHConfig
from structure.store import StructureStore

logger = logging.getLogger(__name__)


@dataclass
class EvidenceResult:
    """Final evidence bundle with retrieval and validation trace."""

    chunks: list[Chunk]
    evidence: EvidenceSet
    adaptive: AdaptiveResult
    initial_validation: ValidationResult | None = None
    validation: ValidationResult | None = None
    retries: list[dict[str, Any]] = field(default_factory=list)

    def trace(self, text_limit: int = 300) -> dict[str, Any]:
        return {
            "controller": {
                "retrievers": self.adaptive.decision.retrievers,
                "k": self.adaptive.decision.k,
                "complexity": self.adaptive.decision.complexity,
                "reason": self.adaptive.decision.reason,
                "signals": self.adaptive.decision.signals,
            },
            "retrieval": {
                "counts": self.adaptive.counts,
                "latencies_ms": self.adaptive.latencies_ms,
                "errors": self.adaptive.errors,
                "selected_sections": self.adaptive.selected_sections,
                "structural_strategy": self.adaptive.structural_strategy,
                "structural_reason": self.adaptive.structural_reason,
            },
            "fused_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "doc": chunk.doc,
                    "page": chunk.page,
                    "score": chunk.score,
                    "sources": chunk.sources,
                    "text": chunk.text[:text_limit],
                }
                for chunk in self.adaptive.chunks
            ],
            "validation": self.validation.to_dict() if self.validation else None,
            "initial_validation": self.initial_validation.to_dict() if self.initial_validation else None,
            "retries": self.retries,
            "evidence": self.evidence.trace(text_limit=text_limit),
        }


def _merge_chunks(primary: list[Chunk], extra: list[Chunk]) -> list[Chunk]:
    merged: dict[str, Chunk] = {}
    for chunk in primary + extra:
        key = chunk.chunk_id or chunk.text[:80]
        if key not in merged:
            merged[key] = chunk
        elif chunk.score > merged[key].score:
            extra_sources = sorted(set(merged[key].sources) | set(chunk.sources))
            chunk.sources = extra_sources
            merged[key] = chunk
        else:
            merged[key].sources = sorted(set(merged[key].sources) | set(chunk.sources))
    return list(merged.values())


class EvidencePipeline:
    """Shared E4/E5 evidence stage; retry mode is a config flag."""

    def __init__(
        self,
        adaptive: AdaptiveRetriever,
        store: StructureStore,
        config: SAHConfig | None = None,
        reranker: CrossEncoderReranker | None = None,
        validator: EvidenceValidator | None = None,
    ):
        self.adaptive = adaptive
        self.store = store
        self.config = config or SAHConfig.from_env()
        self.reranker = reranker or CrossEncoderReranker(self.config.reranker_model)
        self.validator = validator or EvidenceValidator(self.reranker, self.config)
        self.expander = EvidenceExpander(store, self.config)
        self.retrier = EvidenceRetrier(store, self.reranker, self.config)

    def gather(
        self,
        query: str,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> EvidenceResult:
        k = k or self.config.top_k
        adaptive = self.adaptive.retrieve_adaptive(
            query,
            k=k,
            doc_ids=doc_ids,
            fusion_pool=max(k * 2, k) if self.config.rerank else k,
        )
        primary = adaptive.chunks
        if self.config.rerank:
            primary = self.reranker.rerank(query, primary, top_k=k)
        else:
            primary = primary[:k]

        evidence = self.expander.expand(query, primary)
        initial_validation = None
        validation = None
        retries: list[dict[str, Any]] = []

        if self.config.evidence_validation:
            validation = self.validator.validate(query, primary, evidence.context_text)
            initial_validation = validation

            while not validation.sufficient and len(retries) < self.config.max_retries:
                attempt = len(retries) + 1
                if self.config.structure_guided_retry:
                    new_chunks, event = self.retrier.structure_guided(
                        query, validation.missing_aspects, primary, k=k, attempt=attempt
                    )
                else:
                    new_chunks, event = self.retrier.plain(
                        query, validation.missing_aspects, self.adaptive, k=k, doc_ids=doc_ids
                    )

                if not new_chunks:
                    event.update(
                        {
                            "attempt": attempt,
                            "sufficient_after": validation.sufficient,
                            "missing_aspects": validation.missing_aspects,
                            "skipped": True,
                        }
                    )
                    retries.append(event)
                    break

                combined = _merge_chunks(primary, new_chunks)
                if self.config.rerank:
                    combined = self.reranker.rerank(event["query"], combined, top_k=k)
                else:
                    combined.sort(key=lambda chunk: chunk.score, reverse=True)
                    combined = combined[:k]
                primary = combined
                evidence = self.expander.expand(query, primary)
                validation = self.validator.validate(query, primary, evidence.context_text)
                event.update(
                    {
                        "attempt": attempt,
                        "sufficient_after": validation.sufficient,
                        "missing_aspects": validation.missing_aspects,
                        "fused_chunk_ids": [chunk.chunk_id for chunk in primary],
                    }
                )
                retries.append(event)
                logger.info(
                    "Evidence retry %d (%s): sufficient=%s", attempt, event["mode"], validation.sufficient
                )

        return EvidenceResult(
            chunks=primary,
            evidence=evidence,
            adaptive=adaptive,
            initial_validation=initial_validation,
            validation=validation,
            retries=retries,
        )
