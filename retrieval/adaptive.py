"""Parallel adaptive retrieval over semantic, lexical, and structural routes."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from retrieval.baseline import BaselineRetriever
from retrieval.controller import ControllerDecision, QueryController
from retrieval.fusion import rrf_fuse
from retrieval.structural import StructuralRetriever
from retrieval.types import Chunk
from sah_config import SAHConfig

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveResult:
    """Fused chunks plus the controller/retrieval trace."""

    chunks: list[Chunk]
    decision: ControllerDecision
    selected_sections: list[dict[str, Any]] = field(default_factory=list)
    structural_strategy: str = "disabled"
    structural_reason: str = ""
    structural_retries: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    latencies_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class AdaptiveRetriever:
    """Route a query, run selected retrievers concurrently, then RRF-fuse."""

    def __init__(
        self,
        baseline: BaselineRetriever,
        structural: StructuralRetriever | None = None,
        config: SAHConfig | None = None,
        controller: QueryController | None = None,
    ):
        self.baseline = baseline
        self.structural = structural
        self.config = config or SAHConfig.from_env()
        self.controller = controller or QueryController()

    def _route(self, query: str) -> ControllerDecision:
        if self.config.adaptive_controller:
            decision = self.controller.analyze(query, base_k=self.config.top_k)
        else:
            retrievers = ["semantic", "lexical"]
            if self.config.structural_retrieval and self.structural:
                retrievers.append("structural")
            decision = ControllerDecision(
                retrievers=retrievers,
                k=self.config.top_k,
                complexity="fixed",
                reason="Adaptive controller disabled; fixed retriever selection from config.",
                signals={},
            )
        if not self.config.structural_retrieval:
            decision.retrievers = [name for name in decision.retrievers if name != "structural"]
        if not self.structural:
            decision.retrievers = [name for name in decision.retrievers if name != "structural"]
        return decision

    def retrieve_adaptive(
        self,
        query: str,
        k: int | None = None,
        doc_ids: list[str] | None = None,
        fusion_pool: int | None = None,
    ) -> AdaptiveResult:
        decision = self._route(query)
        depth = k or decision.k
        structural_result = None
        errors: dict[str, str] = {}
        latencies: dict[str, float] = {}
        counts: dict[str, int] = {}

        def run_semantic() -> list[Chunk]:
            return self.baseline.retrieve_dense(query, k=depth)

        def run_lexical() -> list[Chunk]:
            return self.baseline.retrieve_sparse(query, k=depth)

        def run_structural() -> Any:
            if not self.structural:
                return None
            return self.structural.retrieve_structural(query, k=depth, doc_ids=doc_ids)

        tasks = {}
        submitted_at = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            if "semantic" in decision.retrievers:
                future = pool.submit(run_semantic)
                tasks[future] = "semantic"
                submitted_at[future] = time.perf_counter()
            if "lexical" in decision.retrievers:
                future = pool.submit(run_lexical)
                tasks[future] = "lexical"
                submitted_at[future] = time.perf_counter()
            if "structural" in decision.retrievers:
                future = pool.submit(run_structural)
                tasks[future] = "structural"
                submitted_at[future] = time.perf_counter()

            result_lists: dict[str, list[Chunk]] = {}
            for future in as_completed(tasks):
                name = tasks[future]
                try:
                    value = future.result()
                except Exception as exc:  # one retriever must not break the request
                    logger.warning("Retriever %s failed: %s", name, exc)
                    errors[name] = str(exc)
                    counts[name] = 0
                    continue
                latencies[name] = (time.perf_counter() - submitted_at[future]) * 1000.0
                if name == "structural":
                    structural_result = value
                    result_lists[name] = value.chunks if value else []
                else:
                    result_lists[name] = value or []
                counts[name] = len(result_lists[name])

        if not result_lists:
            # Defensive fallback for a total semantic failure.
            try:
                result_lists["semantic"] = self.baseline.retrieve_dense(query, k=depth)
                counts["semantic"] = len(result_lists["semantic"])
            except Exception as exc:
                errors["semantic"] = str(exc)

        pool_size = fusion_pool or max(depth, self.config.top_k)
        fused = rrf_fuse(
            result_lists,
            top_k=pool_size,
            weights={"semantic": 1.0, "lexical": 1.0, "structural": 1.15},
        )

        result = AdaptiveResult(
            chunks=fused,
            decision=decision,
            counts=counts,
            latencies_ms=latencies,
            errors=errors,
        )
        if structural_result is not None:
            result.selected_sections = structural_result.selected_sections
            result.structural_strategy = structural_result.strategy
            result.structural_reason = structural_result.reason
            result.structural_retries = structural_result.retries

        logger.info(
            "Adaptive retrieval: retrievers=%s counts=%s fused=%d",
            decision.retrievers, counts, len(fused),
        )
        return result
