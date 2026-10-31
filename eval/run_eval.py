#!/usr/bin/env python3
"""Run retrieval evaluation for SAH-RAG experiments E1-E5.

E1: legacy BM25 + dense + RRF
E2: STAIR-inspired structural retrieval
E3: adaptive controller + three-way RRF
E4: E3 + cross-encoder rerank + context expansion
E5: E4 + cross-encoder validation + structure-guided/plain retry
"""

from __future__ import annotations

import argparse
import json
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


def build_retriever(args: argparse.Namespace):
    """Build an in-process hybrid index and (optionally) the structure index."""
    from core.ingestion.embedder import EmbeddingGenerator
    from core.ingestion.indexer import IndexBuilder
    from core.ingestion.storage import Storage
    from core.retrieval.embedding_model import QueryEmbeddingModel
    from core.retrieval.tfidf_retriever import TFIDFRetriever
    from core.retrieval.vector_db_retriever import VectorDBRetriever
    from structure.ingest import ingest_document
    from structure.store import StructureStore

    storage = Storage(use_pinecone=args.use_pinecone)
    embedder = EmbeddingGenerator()
    builder = IndexBuilder(data_dir=args.data_dir, embedder=embedder, storage=storage)
    chunks_by_file = {}

    if args.ingest:
        data_dir = Path(args.data_dir)
        for path in sorted(data_dir.iterdir()) if data_dir.exists() else []:
            if path.suffix.lower() not in {".pdf", ".txt", ".md", ".markdown", ".docx"}:
                continue
            chunks = builder.build_index_for_file(str(path))
            if chunks:
                chunks_by_file[str(path)] = chunks

    tfidf = TFIDFRetriever(storage=storage)
    vector_db = VectorDBRetriever(storage=storage)
    query_embedder = QueryEmbeddingModel(embedder=embedder)
    structure_store = StructureStore(args.db, args.structure_json_dir)

    if args.experiment in {"E2", "E3", "E4", "E5"} and args.ingest:
        for path, chunks in chunks_by_file.items():
            ingest_document(path, structure_store, existing_chunks=chunks)

    baseline = BaselineRetriever(
        tfidf=tfidf,
        vector_db=vector_db,
        embedder=query_embedder,
        storage=storage,
        structure_store=structure_store,
    )
    return baseline, storage, structure_store, chunks_by_file


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


def _retrieval_row(item: dict[str, Any], experiment: str, chunks, k: int, latency_ms: float) -> dict[str, Any]:
    flags = [is_relevant(chunk, item) for chunk in chunks]
    return {
        "experiment": experiment,
        "question": item["question"],
        "doc": item["doc"],
        "gold_section": item["gold_section"],
        "answerable": bool(item["answerable"]),
        f"recall@{k}": recall_at_k(flags, k),
        f"precision@{k}": precision_at_k(flags, k),
        "mrr": reciprocal_rank(flags),
        f"ndcg@{k}": ndcg_at_k(flags, k),
        "latency_ms": latency_ms,
        "retrieved_ids": ";".join(chunk.chunk_id for chunk in chunks),
        "retrieved_sections": ";".join(chunk.section_title or "" for chunk in chunks),
    }


def _summarize(frame: pd.DataFrame, k: int, extra: dict[str, float] | None = None) -> dict[str, float]:
    metric_cols = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", "latency_ms"]
    summary = {column: float(frame[column].mean()) for column in metric_cols}
    summary.update(extra or {})
    summary["n_queries"] = float(len(frame))
    return summary


def evaluate_e1(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate the baseline retriever (experiment E1)."""
    questions = load_questions(Path(args.questions))
    retriever, _, _, _ = build_retriever(args)
    rows = []
    for item in questions:
        started = time.perf_counter()
        chunks = retriever.retrieve_baseline(item["question"], k=args.k)
        rows.append(_retrieval_row(item, "E1", chunks, args.k, (time.perf_counter() - started) * 1000))
    frame = pd.DataFrame(rows)
    return frame, _summarize(frame, args.k)


def evaluate_e2(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate the STAIR-inspired structure-aware retriever (experiment E2)."""
    from retrieval.structural import StructuralRetriever

    questions = load_questions(Path(args.questions))
    _, _, store, _ = build_retriever(args)
    structural = StructuralRetriever(store)
    rows = []
    for item in questions:
        document = store.get_document_by_filename(item["doc"])
        doc_ids = [document["doc_id"]] if document else None
        started = time.perf_counter()
        result = structural.retrieve_structural(item["question"], k=args.k, doc_ids=doc_ids)
        row = _retrieval_row(item, "E2", result.chunks, args.k, (time.perf_counter() - started) * 1000)
        row.update(
            {
                "selected_sections": ";".join(entry["title"] for entry in result.selected_sections),
                "selection_strategy": result.strategy,
                "selection_retries": result.retries,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, _summarize(frame, args.k)


def evaluate_e3(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate adaptive controller + three-way RRF fusion (experiment E3)."""
    from retrieval.adaptive import AdaptiveRetriever
    from retrieval.structural import StructuralRetriever
    from sah_config import SAHConfig, config_for_experiment

    questions = load_questions(Path(args.questions))
    baseline, _, store, _ = build_retriever(args)
    config = config_for_experiment("E3", SAHConfig.from_env())
    config.top_k = args.k
    adaptive = AdaptiveRetriever(baseline, StructuralRetriever(store, config=config), config=config)
    rows = []
    for item in questions:
        document = store.get_document_by_filename(item["doc"])
        doc_ids = [document["doc_id"]] if document else None
        started = time.perf_counter()
        result = adaptive.retrieve_adaptive(item["question"], k=args.k, doc_ids=doc_ids)
        row = _retrieval_row(item, "E3", result.chunks, args.k, (time.perf_counter() - started) * 1000)
        row.update(
            {
                "controller_retrievers": ",".join(result.decision.retrievers),
                "controller_complexity": result.decision.complexity,
                "controller_reason": result.decision.reason,
                "retriever_counts": json.dumps(result.counts),
                "retriever_latencies_ms": json.dumps(result.latencies_ms),
                "selected_sections": ";".join(entry["title"] for entry in result.selected_sections),
                "structural_strategy": result.structural_strategy,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, _summarize(frame, args.k)


def evaluate_evidence(args: argparse.Namespace, experiment: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate E4 (expanded evidence + rerank) and E5 (validation + retries)."""
    from evidence.pipeline import EvidencePipeline
    from retrieval.adaptive import AdaptiveRetriever
    from retrieval.structural import StructuralRetriever
    from sah_config import SAHConfig, config_for_experiment

    questions = load_questions(Path(args.questions))
    baseline, _, store, _ = build_retriever(args)
    config = config_for_experiment(experiment, SAHConfig.from_env())
    config.top_k = args.k
    if args.retry_mode:
        config.structure_guided_retry = args.retry_mode == "structure"
    adaptive = AdaptiveRetriever(baseline, StructuralRetriever(store, config=config), config=config)
    pipeline = EvidencePipeline(adaptive, store, config=config)
    rows = []
    for item in questions:
        document = store.get_document_by_filename(item["doc"])
        doc_ids = [document["doc_id"]] if document else None
        started = time.perf_counter()
        result = pipeline.gather(item["question"], k=args.k, doc_ids=doc_ids)
        row = _retrieval_row(
            item, experiment, result.chunks, args.k, (time.perf_counter() - started) * 1000
        )
        validation = result.validation
        row.update(
            {
                "controller_retrievers": ",".join(result.adaptive.decision.retrievers),
                "controller_complexity": result.adaptive.decision.complexity,
                "controller_reason": result.adaptive.decision.reason,
                "selected_sections": ";".join(
                    entry["title"] for entry in result.adaptive.selected_sections
                ),
                "structural_strategy": result.adaptive.structural_strategy,
                "retry_count": len(result.retries),
                "retry_modes": ",".join(event.get("mode", "") for event in result.retries),
                "retry_events": json.dumps(result.retries),
                "validation_sufficient": validation.sufficient if validation else None,
                "validation_max_score": validation.max_score if validation else None,
                "validation_coverage": validation.coverage if validation else None,
                "validation_missing": ";".join(validation.missing_aspects) if validation else "",
                "evidence_tokens": result.evidence.token_count,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    extra = {"retry_count": float(frame["retry_count"].mean())} if "retry_count" in frame else {}
    return frame, _summarize(frame, args.k, extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAH-RAG retrieval evaluation harness")
    parser.add_argument("--experiment", default="E1", choices=["E1", "E2", "E3", "E4", "E5"])
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ingest", action="store_true", help="Rebuild the in-memory index from data/")
    parser.add_argument("--use-pinecone", action="store_true")
    parser.add_argument("--db", default=str(ROOT / "data" / "sah_rag.db"))
    parser.add_argument("--structure-json-dir", default=str(ROOT / "data" / "structure"))
    parser.add_argument("--retry-mode", choices=["structure", "plain"], default=None)
    args = parser.parse_args()

    if args.experiment == "E1":
        frame, summary = evaluate_e1(args)
    elif args.experiment == "E2":
        frame, summary = evaluate_e2(args)
    elif args.experiment == "E3":
        frame, summary = evaluate_e3(args)
    elif args.experiment in {"E4", "E5"}:
        frame, summary = evaluate_evidence(args, args.experiment)
    else:
        raise SystemExit(f"Unknown experiment: {args.experiment}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.retry_mode is None or args.experiment != "E5" else f"_{args.retry_mode}"
    per_query_path = output_dir / f"{args.experiment}{suffix}_per_question.csv"
    summary_path = output_dir / f"{args.experiment}{suffix}_summary.csv"
    frame.to_csv(per_query_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    pd.set_option("display.width", 240)
    print("\nPer-question results")
    print(frame.to_string(index=False))
    print("\nSummary")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved: {per_query_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
