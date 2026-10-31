"""Retrieval and ranking metrics used by every experiment."""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(relevant: Iterable[bool], k: int) -> float:
    """Recall@K for a single query with binary relevance labels."""
    flags = list(relevant)[:k]
    total_relevant = sum(1 for flag in relevant if flag)
    if total_relevant == 0:
        return 0.0
    return sum(1 for flag in flags if flag) / total_relevant


def precision_at_k(relevant: Iterable[bool], k: int) -> float:
    """Precision@K with K in the denominator (standard IR definition)."""
    flags = list(relevant)[:k]
    if k <= 0:
        return 0.0
    return sum(1 for flag in flags if flag) / k


def reciprocal_rank(relevant: Iterable[bool]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, flag in enumerate(relevant, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Iterable[bool], k: int) -> float:
    """Binary-gain nDCG@K."""
    flags = list(relevant)[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(flags, start=1) if flag)
    ideal = sorted((1 for flag in relevant if flag), reverse=True)[:k]
    idcg = sum(1.0 / math.log2(rank + 1) for rank, _ in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Simple whitespace token F1 for answer-correctness checks."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    remaining = list(ref_tokens)
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate(rows: list[dict], keys: list[str]) -> dict[str, float]:
    """Mean of numeric metric columns."""
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = sum(values) / len(values) if values else 0.0
    return out
