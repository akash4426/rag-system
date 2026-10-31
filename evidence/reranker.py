"""Cross-encoder reranking for evidence (E4)."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from retrieval.types import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy MiniLM cross-encoder reranker with a graceful fallback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer: Callable[[list[list[str]]], Sequence[float]] | None = None,
    ):
        self.model_name = model_name
        self._scorer = scorer
        self._model = None
        self._failed = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._failed or self._scorer is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # keep retrieval usable without the model
            logger.warning("Cross-encoder unavailable (%s); falling back to retriever scores.", exc)
            self._failed = True

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return raw relevance logits for each text."""
        if not texts:
            return []
        self._ensure_model()
        if self._scorer is not None:
            try:
                return [float(value) for value in self._scorer([[query, text] for text in texts])]
            except Exception as exc:
                logger.warning("Injected cross-encoder scorer failed: %s", exc)
                return [0.0 for _ in texts]
        if self._model is None:
            return [0.0 for _ in texts]
        try:
            values = self._model.predict([[query, text] for text in texts])
            return [float(value) for value in values]
        except Exception as exc:
            logger.warning("Cross-encoder prediction failed: %s", exc)
            return [0.0 for _ in texts]

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Sort chunks by cross-encoder score and keep the top K."""
        if not chunks:
            return []
        scores = self.score(query, [chunk.text for chunk in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"] = score
            chunk.metadata["pre_rerank_score"] = chunk.score
            chunk.score = score
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]
