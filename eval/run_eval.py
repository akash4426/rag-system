#!/usr/bin/env python3
"""Run retrieval and end-to-end evaluation for SAH-RAG experiments E1-E5.

E1: legacy BM25 + dense + RRF
E2: STAIR-inspired structural retrieval
E3: adaptive controller + three-way RRF
E4: E3 + cross-encoder rerank + context expansion
E5: E4 + cross-encoder validation + structure-guided/plain retry

Retrieval-only mode preserves the original E1-E5 metrics. ``--answer-eval``
adds grounded answers, claim-level faithfulness, citation accuracy, abstention,
latency, and token usage. ``--experiment all`` writes a comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
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

EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5"]


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

    if args.experiment in {"E2", "E3", "E4", "E5", "all"} and args.ingest:
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


# --------------------------------------------------------------------- E1-E5 retrieval
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


def evaluate_retrieval(
    args: argparse.Namespace, experiment: str, retry_mode: str | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Dispatch one retrieval-only experiment."""
    previous = args.retry_mode
    args.retry_mode = retry_mode
    try:
        if experiment == "E1":
            return evaluate_e1(args)
        if experiment == "E2":
            return evaluate_e2(args)
        if experiment == "E3":
            return evaluate_e3(args)
        return evaluate_evidence(args, experiment)
    finally:
        args.retry_mode = previous


# --------------------------------------------------------------------- answer-level helpers
def _tokens(text: str) -> list[str]:
    cleaned = re.sub(r"\[(?:E\d+[\s,;]*)\]", " ", text or "")
    return re.findall(r"[a-z0-9]+", cleaned.lower())


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token F1 with citations removed."""
    predicted = _tokens(prediction)
    expected = _tokens(gold)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity with a safe NaN for empty vectors."""
    if left is None or right is None or len(left) == 0 or len(right) == 0:
        return float("nan")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
    right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        return float("nan")
    return dot / (left_norm * right_norm)


def _answer_similarity(embedder, answer: str, gold: str) -> float:
    if not answer.strip() or not gold.strip():
        return float("nan")
    try:
        return cosine_similarity(embedder.embed_query(answer), embedder.embed_query(gold))
    except Exception:
        return float("nan")


def _evidence_map(evidence) -> dict[str, str]:
    return {
        str(item.metadata["evidence_id"]): item.text
        for item in evidence.items
        if item.metadata.get("evidence_id")
    }


def _claim_metrics(report) -> dict[str, float | int]:
    claims = report.claims if report else []
    total = len(claims)
    with_citations = sum(1 for claim in claims if claim.citations)
    supported = sum(1 for claim in claims if claim.supported)
    references = [evidence_id for claim in claims for evidence_id in claim.citations]
    invalid = set(report.invalid_citations if report else [])
    valid_references = sum(1 for evidence_id in references if evidence_id not in invalid)
    return {
        "claims_total": total,
        "claims_supported": supported,
        "claims_with_citations": with_citations,
        "citation_references": len(references),
        "faithfulness": supported / total if total else float("nan"),
        "citation_accuracy": supported / with_citations if with_citations else float("nan"),
        "citation_precision": valid_references / len(references) if references else float("nan"),
        "citation_coverage": with_citations / total if total else float("nan"),
    }


def evaluate_answer(args: argparse.Namespace, experiment: str, retry_mode: str | None = None):
    """Run end-to-end answer evaluation for one experiment."""
    from evidence.expander import EvidenceExpander
    from evidence.pipeline import EvidencePipeline
    from evidence.types import EvidenceSet, count_tokens
    from generation.answerer import GroundedAnswerer
    from generation.generator import GroundedGenerator
    from generation.types import VerificationReport
    from generation.verifier import CitationVerifier
    from retrieval.adaptive import AdaptiveRetriever
    from retrieval.structural import StructuralRetriever
    from sah_config import SAHConfig, config_for_experiment

    questions = load_questions(Path(args.questions))
    baseline, _, store, _ = build_retriever(args)
    config = config_for_experiment(experiment, SAHConfig.from_env())
    config.top_k = args.k
    config.grounded_generation = True
    if experiment == "E5":
        config.structure_guided_retry = (retry_mode or "structure") != "plain"

    expander = EvidenceExpander(store, config)
    reference_verifier = CitationVerifier(config)
    generator = GroundedGenerator(config)
    answerer = GroundedAnswerer(config, generator=generator, verifier=reference_verifier)
    adaptive = AdaptiveRetriever(baseline, StructuralRetriever(store, config=config), config=config)
    pipeline = EvidencePipeline(adaptive, store, config=config)

    rows = []
    for item in questions:
        question = item["question"]
        document = store.get_document_by_filename(item["doc"])
        doc_ids = [document["doc_id"]] if document else None
        started = time.perf_counter()

        result = None
        validation = None
        retries: list[dict[str, Any]] = []
        adaptive_result = None

        retrieval_started = time.perf_counter()
        if experiment == "E1":
            chunks = baseline.retrieve_baseline(question, k=args.k)
            evidence = expander.expand(question, chunks)
        elif experiment == "E2":
            structural = StructuralRetriever(store, config=config)
            structural_result = structural.retrieve_structural(question, k=args.k, doc_ids=doc_ids)
            chunks = structural_result.chunks
            evidence = expander.expand(question, chunks)
        else:
            result = pipeline.gather(question, k=args.k, doc_ids=doc_ids)
            chunks = result.chunks
            evidence = result.evidence
            validation = result.validation
            retries = result.retries
            adaptive_result = result.adaptive
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        prompt = generator.build_prompt(question, evidence)
        prompt_tokens = count_tokens(prompt)

        generation_started = time.perf_counter()
        grounded = answerer.answer(question, evidence, validation)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        verification_started = time.perf_counter()
        if grounded.verification is not None:
            report = grounded.verification
        else:
            report = reference_verifier.verify(grounded.claims, _evidence_map(evidence))
        verification_ms = (time.perf_counter() - verification_started) * 1000

        answer_text = grounded.answer
        answer_clean = re.sub(r"\[(?:E\d+[\s,;]*)\]", " ", answer_text).strip()
        gold = str(item.get("gold_answer", ""))
        answerable = bool(item["answerable"])
        metrics = _claim_metrics(report)

        total_ms = (time.perf_counter() - started) * 1000
        row = _retrieval_row(item, experiment, chunks, args.k, total_ms)
        row.update(
            {
                "retry_mode": retry_mode or ("structure" if experiment == "E5" else "n/a"),
                "retry_count": len(retries),
                "retry_modes": ",".join(event.get("mode", "") for event in retries),
                "abstained": bool(grounded.abstained),
                "abstention_reason": grounded.abstention_reason,
                "answer_f1": token_f1(answer_clean, gold) if answerable else 0.0,
                "answer_similarity": (
                    _answer_similarity(baseline.embedder, answer_clean, gold)
                    if answerable and not grounded.abstained
                    else float("nan")
                ),
                "abstention_correct": (
                    int(grounded.abstained) if not answerable else int(not grounded.abstained)
                ),
                "prompt_tokens": prompt_tokens,
                "answer_tokens": count_tokens(answer_text),
                "evidence_tokens": evidence.token_count,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "verification_ms": verification_ms,
                "answer_text": answer_text[:1200],
                "generated_claims": grounded.raw_response[:1200],
                **metrics,
            }
        )
        if adaptive_result is not None:
            row.update(
                {
                    "controller_retrievers": ",".join(adaptive_result.decision.retrievers),
                    "controller_complexity": adaptive_result.decision.complexity,
                    "selected_sections": ";".join(
                        entry["title"] for entry in adaptive_result.selected_sections
                    ),
                    "structural_strategy": adaptive_result.structural_strategy,
                }
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    answerable_frame = frame[frame["answerable"]]
    unanswerable_frame = frame[~frame["answerable"]]
    extra = {
        "answer_f1": float(answerable_frame["answer_f1"].mean()) if len(answerable_frame) else float("nan"),
        "answer_similarity": float(answerable_frame["answer_similarity"].mean()) if len(answerable_frame) else float("nan"),
        "faithfulness": float(answerable_frame["faithfulness"].mean()) if len(answerable_frame) else float("nan"),
        "citation_accuracy": float(answerable_frame["citation_accuracy"].mean()) if len(answerable_frame) else float("nan"),
        "citation_precision": float(answerable_frame["citation_precision"].mean()) if len(answerable_frame) else float("nan"),
        "citation_coverage": float(answerable_frame["citation_coverage"].mean()) if len(answerable_frame) else float("nan"),
        "decision_accuracy": float(frame["abstention_correct"].mean()),
        "false_abstention_rate": float(answerable_frame["abstained"].mean()) if len(answerable_frame) else float("nan"),
        "unanswerable_abstention_rate": (
            float(unanswerable_frame["abstained"].mean()) if len(unanswerable_frame) else float("nan")
        ),
        "retry_count": float(frame["retry_count"].mean()),
        "retrieval_ms": float(frame["retrieval_ms"].mean()),
        "generation_ms": float(frame["generation_ms"].mean()),
        "verification_ms": float(frame["verification_ms"].mean()),
        "answer_tokens": float(frame["answer_tokens"].mean()),
        "prompt_tokens": float(frame["prompt_tokens"].mean()),
        "evidence_tokens": float(frame["evidence_tokens"].mean()),
    }
    return frame, _summarize(frame, args.k, extra)


# --------------------------------------------------------------------- output
def _result_key(experiment: str, retry_mode: str | None) -> str:
    if experiment != "E5":
        return experiment
    return f"E5_{'plain' if retry_mode == 'plain' else 'structure'}"


def _suffix(experiment: str, retry_mode: str | None) -> str:
    if experiment != "E5":
        return ""
    return "_plain" if retry_mode == "plain" else "_structure"


def save_results(
    output_dir: Path,
    experiment: str,
    frame: pd.DataFrame,
    summary: dict[str, float],
    retry_mode: str | None = None,
    answer_level: bool = False,
) -> Path:
    """Write per-question and summary CSVs and return the summary path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(experiment, retry_mode)
    kind = "_answer" if answer_level else ""
    frame.to_csv(output_dir / f"{experiment}{suffix}{kind}_per_question.csv", index=False)
    summary_path = output_dir / f"{experiment}{suffix}{kind}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary_path


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number):
        return "—"
    return f"{number:.{digits}f}"


def write_comparison(output_dir: Path, summaries: dict[str, dict[str, float]]) -> None:
    """Write CSV and Markdown comparison tables in a stable experiment order."""
    order = ["E1", "E2", "E3", "E4", "E5_structure", "E5_plain"]
    keys = [key for key in order if key in summaries] + sorted(set(summaries) - set(order))
    rows = []
    for key in keys:
        summary = summaries[key]
        rows.append(
            {
                "experiment": key,
                "recall@k": summary.get("recall@5"),
                "precision@k": summary.get("precision@5"),
                "mrr": summary.get("mrr"),
                "ndcg@k": summary.get("ndcg@5"),
                "answer_f1": summary.get("answer_f1"),
                "answer_similarity": summary.get("answer_similarity"),
                "faithfulness": summary.get("faithfulness"),
                "citation_accuracy": summary.get("citation_accuracy"),
                "citation_precision": summary.get("citation_precision"),
                "citation_coverage": summary.get("citation_coverage"),
                "decision_accuracy": summary.get("decision_accuracy"),
                "unanswerable_abstention_rate": summary.get("unanswerable_abstention_rate"),
                "false_abstention_rate": summary.get("false_abstention_rate"),
                "retry_count": summary.get("retry_count"),
                "latency_ms": summary.get("latency_ms"),
                "retrieval_ms": summary.get("retrieval_ms"),
                "generation_ms": summary.get("generation_ms"),
                "answer_tokens": summary.get("answer_tokens"),
                "prompt_tokens": summary.get("prompt_tokens"),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "comparison.csv", index=False)

    headers = [
        "Experiment", "R@5", "P@5", "MRR", "nDCG@5", "Answer F1", "Answer sim",
        "Faithful", "Citation acc", "Citation cov", "Decision acc",
        "Unans. abstain", "False abstain", "Retries", "Total ms", "Ans tokens",
    ]
    lines = [
        "# SAH-RAG Experiment Comparison",
        "",
        "Retrieval and answer metrics produced by `eval/run_eval.py`.",
        "Answer metrics are averaged over answerable questions unless noted.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    _fmt(row["recall@k"]),
                    _fmt(row["precision@k"]),
                    _fmt(row["mrr"]),
                    _fmt(row["ndcg@k"]),
                    _fmt(row["answer_f1"]),
                    _fmt(row["answer_similarity"]),
                    _fmt(row["faithfulness"]),
                    _fmt(row["citation_accuracy"]),
                    _fmt(row["citation_coverage"]),
                    _fmt(row["decision_accuracy"]),
                    _fmt(row["unanswerable_abstention_rate"]),
                    _fmt(row["false_abstention_rate"]),
                    _fmt(row["retry_count"], 2),
                    _fmt(row["latency_ms"], 0),
                    _fmt(row["answer_tokens"], 0),
                ]
            )
            + " |"
        )
    lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {output_dir / 'comparison.csv'}")
    print(f"Saved: {output_dir / 'comparison.md'}")
    print("\n" + "\n".join(lines))


def load_saved_summaries(output_dir: Path) -> dict[str, dict[str, float]]:
    """Load answer summaries (or retrieval summaries) without rerunning models."""
    files = {
        "E1": ("E1_answer_summary.csv", "E1_summary.csv"),
        "E2": ("E2_answer_summary.csv", "E2_summary.csv"),
        "E3": ("E3_answer_summary.csv", "E3_summary.csv"),
        "E4": ("E4_answer_summary.csv", "E4_summary.csv"),
        "E5_structure": ("E5_structure_answer_summary.csv", "E5_structure_summary.csv"),
        "E5_plain": ("E5_plain_answer_summary.csv", "E5_plain_summary.csv"),
    }
    summaries: dict[str, dict[str, float]] = {}
    for key, candidates in files.items():
        for filename in candidates:
            path = output_dir / filename
            if path.exists():
                summaries[key] = pd.read_csv(path).iloc[0].to_dict()
                break
    return summaries


def run_all(args: argparse.Namespace) -> None:
    """Run E1-E5 retrieval, optional answer evaluation, and the E5 ablation."""
    output_dir = Path(args.output_dir)
    retrieval_summaries: dict[str, dict[str, float]] = {}

    for experiment in EXPERIMENTS:
        retry_mode = "structure" if experiment == "E5" else None
        frame, summary = evaluate_retrieval(args, experiment, retry_mode=retry_mode)
        key = _result_key(experiment, retry_mode)
        retrieval_summaries[key] = summary
        save_results(output_dir, experiment, frame, summary, retry_mode=retry_mode)
        print(f"{key}: R@5={summary.get('recall@5'):.3f} MRR={summary.get('mrr'):.3f}")

    if not args.answer_eval:
        write_comparison(output_dir, retrieval_summaries)
        return

    answer_summaries: dict[str, dict[str, float]] = {}
    for experiment in EXPERIMENTS:
        retry_mode = "structure" if experiment == "E5" else None
        frame, summary = evaluate_answer(args, experiment, retry_mode=retry_mode)
        key = _result_key(experiment, retry_mode)
        answer_summaries[key] = summary
        save_results(
            output_dir, experiment, frame, summary, retry_mode=retry_mode, answer_level=True
        )
        print(
            f"{key}: F1={summary.get('answer_f1'):.3f} "
            f"faithful={summary.get('faithfulness'):.3f} "
            f"abstain={summary.get('unanswerable_abstention_rate'):.3f}"
        )

    frame, summary = evaluate_answer(args, "E5", retry_mode="plain")
    answer_summaries["E5_plain"] = summary
    save_results(output_dir, "E5", frame, summary, retry_mode="plain", answer_level=True)
    print(
        f"E5_plain: F1={summary.get('answer_f1'):.3f} "
        f"faithful={summary.get('faithfulness'):.3f} "
        f"abstain={summary.get('unanswerable_abstention_rate'):.3f}"
    )

    write_comparison(output_dir, answer_summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAH-RAG retrieval and end-to-end evaluation")
    parser.add_argument(
        "--experiment", default="E1", choices=["E1", "E2", "E3", "E4", "E5", "all"]
    )
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ingest", action="store_true", help="Rebuild the in-memory index from data/")
    parser.add_argument("--use-pinecone", action="store_true")
    parser.add_argument("--db", default=str(ROOT / "data" / "sah_rag.db"))
    parser.add_argument("--structure-json-dir", default=str(ROOT / "data" / "structure"))
    parser.add_argument("--retry-mode", choices=["structure", "plain"], default=None)
    parser.add_argument(
        "--answer-eval",
        action="store_true",
        help="Generate grounded answers and compute answer/faithfulness/citation metrics",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Rebuild comparison.csv/comparison.md from saved summary CSVs",
    )
    args = parser.parse_args()

    if args.compare_only:
        summaries = load_saved_summaries(Path(args.output_dir))
        if not summaries:
            raise SystemExit(f"No summary CSVs found in {args.output_dir}")
        write_comparison(Path(args.output_dir), summaries)
        return

    if args.experiment == "all":
        run_all(args)
        return

    if args.answer_eval:
        frame, summary = evaluate_answer(args, args.experiment, retry_mode=args.retry_mode)
        summary_path = save_results(
            Path(args.output_dir),
            args.experiment,
            frame,
            summary,
            retry_mode=args.retry_mode,
            answer_level=True,
        )
        write_comparison(
            Path(args.output_dir), {_result_key(args.experiment, args.retry_mode): summary}
        )
    else:
        frame, summary = evaluate_retrieval(args, args.experiment, retry_mode=args.retry_mode)
        summary_path = save_results(
            Path(args.output_dir),
            args.experiment,
            frame,
            summary,
            retry_mode=args.retry_mode,
        )

    pd.set_option("display.width", 260)
    print("\nPer-question results")
    print(frame.to_string(index=False))
    print("\nSummary")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
