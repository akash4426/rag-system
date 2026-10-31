"""Phase 8 tests: answer metrics and comparison output."""

from __future__ import annotations

import math

from eval.run_eval import (
    _claim_metrics,
    _result_key,
    _suffix,
    cosine_similarity,
    token_f1,
    write_comparison,
)
from generation.types import Claim, VerificationReport


def test_token_f1_ignores_citation_markers():
    assert token_f1("Three-address code [E1]", "three address code") == 1.0
    assert token_f1("unrelated answer [E2]", "three address code") == 0.0
    assert token_f1("", "gold") == 0.0


def test_cosine_similarity_handles_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([], [1.0]) != cosine_similarity([], [1.0])  # NaN
    assert math.isclose(cosine_similarity([1.0, 1.0], [1.0, 0.0]), 0.707106, rel_tol=1e-5)


def test_claim_metrics_separate_coverage_and_accuracy():
    report = VerificationReport(
        claims=[
            Claim("supported [E1]", ["E1"], supported=True),
            Claim("invalid cite [E9]", ["E9"], supported=False),
            Claim("no citation", [], supported=False),
        ],
        invalid_citations=["E9"],
    )
    metrics = _claim_metrics(report)
    assert metrics["claims_total"] == 3
    assert metrics["claims_supported"] == 1
    assert metrics["faithfulness"] == 1 / 3
    assert metrics["citation_accuracy"] == 1 / 2
    assert metrics["citation_precision"] == 1 / 2
    assert metrics["citation_coverage"] == 2 / 3


def test_e5_result_keys_and_suffixes():
    assert _result_key("E1", None) == "E1"
    assert _result_key("E5", None) == "E5_structure"
    assert _result_key("E5", "plain") == "E5_plain"
    assert _suffix("E1", None) == ""
    assert _suffix("E5", None) == "_structure"
    assert _suffix("E5", "structure") == "_structure"
    assert _suffix("E5", "plain") == "_plain"


def test_write_comparison_creates_csv_and_markdown(tmp_path):
    summaries = {
        "E1": {
            "recall@5": 0.5, "mrr": 0.4, "ndcg@5": 0.45,
            "answer_f1": 0.3, "answer_similarity": 0.7, "faithfulness": 0.6,
            "citation_accuracy": 0.5, "citation_coverage": 0.9,
            "decision_accuracy": 1.0, "unanswerable_abstention_rate": 1.0,
            "false_abstention_rate": 0.0, "retry_count": 0.0,
            "latency_ms": 1000.0, "answer_tokens": 40.0,
        }
    }
    write_comparison(tmp_path, summaries)
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "| E1 |" in markdown
    assert (tmp_path / "comparison.csv").exists()
