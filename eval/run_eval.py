#!/usr/bin/env python3
"""Run retrieval/answer evaluation for SAH-RAG experiments.

Phase 1 implements E1 (legacy BM25 + dense + RRF). Later phases extend this
file with E2-E5 and answer-level metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from retrieval.baseline import BaselineRetriever  # noqa: E402


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the evaluation question file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    required = {"question", "doc", "gold_section", "answerable"}
    for index, row in enumerate(data):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Question {index} is missing fields: {sorted(missing)}")
    return data


def build_retriever(args: argparse.Namespace) -> tuple[BaselineRetriever, Any]:
    """Build an in-process baseline index (or use the configured vector store)."""
    from core.ingestion.embedder import EmbeddingGenerator
    from core.ingestion.indexer import IndexBuilder
    from core.ingestion.storage import Storage
    from core.retrieval.embedding_model import QueryEmbeddingModel
    from core.retrieval.tfidf_retriever import TFIDFRetriever
    from core.retrieval.vector_db_retriever import VectorDBRetriever

    storage = Storage(use_pinecone=args.use_pinecone)
    embedder = EmbeddingGenerator()
    if args.ingest:
        builder = IndexBuilder(data_dir=args.data_dir, embedder=embedder, storage=storage)
        builder.build_index()

    tfidf = TFIDFRetriever(storage=storage)
    vector_db = VectorDBRetriever(storage=storage)
    query_embedder = QueryEmbeddingModel(embedder=embedder)
    return BaselineRetriever(tfidf=tfidf, vector_db=vector_db, embedder=query_embedder, storage=storage), storage


def is_relevant(chunk, question: dict[str, Any]) -> bool:
    """Gold-section relevance with a transparent text fallback."""
    if not question.get("answerable", True):
        return False
    gold = str(question.get("gold_section", "")).strip().lower()
    if not gold or gold == "none":
        return False
    if chunk.section_title and gold in chunk.section_title.lower():
        return True
    if chunk.section_id and gold == str(chunk.section_id).lower():
        return True
    return gold in (chunk.text or "").lower()


def evaluate_e1(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate the baseline retriever (experiment E1)."""
    questions = load_questions(Path(args.questions))
    retriever, _ = build_retriever(args)
    rows: list[dict[str, Any]] = []

    for item in questions:
        started = time.perf_counter()
        chunks = retriever.retrieve_baseline(item["question"], k=args.k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        flags = [is_relevant(chunk, item) for chunk in chunks]
        rows.append(
            {
                "experiment": "E1",
                "question": item["question"],
                "doc": item["doc"],
                "gold_section": item["gold_section"],
                "answerable": bool(item["answerable"]),
                f"recall@{args.k}": recall_at_k(flags, args.k),
                f"precision@{args.k}": precision_at_k(flags, args.k),
                "mrr": reciprocal_rank(flags),
                f"ndcg@{args.k}": ndcg_at_k(flags, args.k),
                "latency_ms": latency_ms,
                "retrieved_ids": ";".join(chunk.chunk_id for chunk in chunks),
                "retrieved_sections": ";".join(chunk.section_title or "" for chunk in chunks),
            }
        )

    frame = pd.DataFrame(rows)
    metric_cols = [f"recall@{args.k}", f"precision@{args.k}", "mrr", f"ndcg@{args.k}", "latency_ms"]
    summary = {column: float(frame[column].mean()) for column in metric_cols}
    summary["n_queries"] = float(len(frame))
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SAH-RAG evaluation harness")
    parser.add_argument("--experiment", default="E1", choices=["E1", "E2", "E3", "E4", "E5"])
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ingest", action="store_true", help="Rebuild the in-memory index from data/")
    parser.add_argument("--use-pinecone", action="store_true", help="Use configured Pinecone instead of in-memory storage")
    args = parser.parse_args()

    if args.experiment != "E1":
        raise SystemExit(f"{args.experiment} is not implemented until its phase is complete")

    frame, summary = evaluate_e1(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / f"{args.experiment}_per_question.csv"
    summary_path = output_dir / f"{args.experiment}_summary.csv"
    frame.to_csv(per_query_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    print("\nPer-question results")
    print(frame.to_string(index=False))
    print("\nSummary")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved: {per_query_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
