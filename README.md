# SAH-RAG — Structure-Aware Adaptive Hybrid RAG

> **23-component retrieval-augmented generation system** with STAIR-inspired structural retrieval, adaptive query routing, grounded citation generation, and a live SSE trace UI.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [STAIR-Inspired Structural Retrieval](#stair-inspired-structural-retrieval)
4. [SAH Pipeline Stages](#sah-pipeline-stages)
5. [Live Trace UI](#live-trace-ui)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)
8. [Quick Start](#quick-start)
9. [API Reference](#api-reference)
10. [Environment Variables](#environment-variables)
11. [Feature Flags & Experiments](#feature-flags--experiments)
12. [Troubleshooting](#troubleshooting)

---

## Overview

SAH-RAG is a research-grade RAG system built on top of a 23-component baseline pipeline. It adds a suite of opt-in enhancements — collectively called the **SAH stack** — each switchable independently through environment flags:

| Layer | Capability |
|---|---|
| **E1** | Baseline BM25 + dense vector hybrid retrieval |
| **E2** | Structure-aware retrieval (STAIR-inspired section selection) |
| **E3** | Adaptive query controller (routes by query complexity) |
| **E4** | Cross-encoder reranking + evidence expansion |
| **E5** | Evidence validation + grounded generation + citation verification |

All SAH features are **additive** — disabling them falls back to the untouched E1 baseline path.

---

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Processing (23-component baseline)                   │
│  Preprocess → Intent Classify → Query Expand                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Adaptive Controller  (E3)                                  │
│  Rule-based signal analysis → selects retrievers + depth k  │
│  Complexity: simple / moderate / complex                    │
└──────┬───────────────┬──────────────────┬───────────────────┘
       │               │                  │
       ▼               ▼                  ▼
  Semantic         Lexical          Structural (E2)
  (dense)          (BM25)       ┌───────────────────────┐
  Pinecone         TF-IDF       │  STAIR-Inspired Path  │
  all-MiniLM-L6-v2             │  ToC render           │
                                │  LLM section select   │
                                │  (gemma3:4b / Ollama) │
                                │  keyword fallback     │
                                │  Section → chunks     │
                                └───────────────────────┘
       │               │                  │
       └───────────────┴──────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Reciprocal Rank Fusion  (E3)                               │
│  Weighted RRF  {semantic: 1.0, lexical: 1.0, structural: 1.15} │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Cross-Encoder Reranker  (E4)                               │
│  ms-marco-MiniLM-L-6-v2 · MPS accelerated                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Evidence Expansion + Validation  (E4 / E5)                 │
│  Token budget packing · cross-encoder sufficiency check     │
│  Missing-aspect retry (structure-guided or plain)           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Grounded Generation  (E5)                                  │
│  Ollama (gemma3:4b) · citation-first prompt                 │
│  Abstain sentinel · claim splitting · citation verification │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
                    Answer + Citations
```

---

## STAIR-Inspired Structural Retrieval

> **STAIR** stands for **S**tructure-**T**oC-**A**ware **I**ndexed **R**etrieval.  
> This implementation is inspired by the idea of LLM-guided document navigation; it is not a reproduction of any published paper.

### Core Idea

Instead of treating a document as a flat bag of chunks, the system first extracts its **section hierarchy** at upload time, then uses a local LLM to **navigate the table of contents** and select the most relevant sections before expanding them to chunks.

This yields structurally coherent retrieval — the system asks *"which section talks about X?"* before pulling text, rather than blindly searching over all chunks.

### How It Works — Step by Step

```
Upload time (once):
  Document → Structure Extractor → Section tree + chunks → SQLite (sah_rag.db)

Query time (per request):
  1. Query Controller decides to include "structural" retriever
  2. StructuralRetriever renders a compact ToC from the section tree
  3. Local Ollama (gemma3:4b) reads the ToC and returns JSON:
       {"section_ids": ["doc::s0012", "doc::s0019"]}
  4. IDs are validated against the real leaf section map
  5. If LLM returns invalid JSON or no IDs → keyword fallback over titles
  6. Selected sections expand into their stored chunks (rank-score weighted)
  7. Structural chunks enter the RRF pool alongside semantic + lexical results
```

### Structure Extraction Strategies

The `StructureExtractor` tries strategies in priority order per format:

| Format | Strategy 1 (preferred) | Strategy 2 | Strategy 3 |
|--------|------------------------|------------|------------|
| **PDF** | PDF bookmarks / ToC | Font-size + bold heading detection | Fixed-size chunks |
| **DOCX** | Word heading styles (Heading 1–6) | Body text paragraphs | Fixed-size chunks |
| **Markdown** | ATX headings (`#` – `######`) | Text-pattern headings | Fixed-size chunks |
| **TXT** | Chapter/Section/Part patterns | — | Fixed-size chunks |

### Section Selection — LLM Prompt

```
/no_think
You navigate a document by its table of contents.

Select up to {top_n} leaf sections most likely to contain the answer.
Return JSON only, exactly this shape:
{"section_ids": ["<id>", "<id>"]}

Rules:
- Use IDs copied exactly from the table of contents.
- Choose only entries marked [leaf].
- Do not invent IDs.
- Return an empty list if nothing is relevant.
- No prose, no Markdown.

QUESTION:
{query}

TABLE OF CONTENTS:
{toc}
```

### Keyword Fallback

When Ollama is unavailable, the system falls back to a **deterministic keyword scorer** that matches query terms against section titles and body text (with stopword filtering), returning the top-N most relevant sections. The UI clearly flags this as `keyword_fallback`.

### Auto-Recovery

After an Ollama outage, `StructuralRetriever` re-checks availability every **30 seconds** instead of permanently caching the `unavailable` state. The system self-heals without a server restart.

### RRF Weight Boost

Structural chunks receive a **1.15× weight** in reciprocal rank fusion vs. 1.0 for semantic and lexical. This reflects their higher precision — the LLM actively chose those sections.

---

## SAH Pipeline Stages

### Stage E2 — Structure-Aware Retrieval

**Files:** `retrieval/structural.py`, `structure/extractor.py`, `structure/store.py`, `structure/ingest.py`

- Document section trees are stored in `data/sah_rag.db` (SQLite) alongside a JSON mirror in `data/structure/`
- Each chunk is linked to its parent section via `section_id`, enabling reverse lookup and expansion
- `StructureStore` supports: `get_leaf_sections()`, `get_chunks_for_sections()`, `find_chunk_by_text()`

### Stage E3 — Adaptive Query Routing

**File:** `retrieval/controller.py`

Signal-based rule engine, no learned classifier:

| Signal | Complexity | Retrievers | Depth k |
|--------|-----------|-----------|---------|
| Compare / multi-entity terms | `complex` | semantic + lexical + structural | `max(k×2, 8)` |
| Explicit section / chapter reference | `moderate` | structural + semantic | `max(k, 5)` |
| Long query (≥28 words) or explain terms | `complex` | semantic + lexical + structural | `max(k×2, 8)` |
| Short factual (≤8 words) | `simple` | semantic + lexical | `k` |
| Default | `moderate` | semantic + lexical | `k` |

When `SAH_ADAPTIVE_CONTROLLER=false`, retriever selection is fixed by config.

### Stage E3 — Parallel Execution & RRF

**File:** `retrieval/adaptive.py`

All three retrievers run in a `ThreadPoolExecutor(max_workers=3)`. Results are collected via `as_completed()` and fused with weighted RRF (`retrieval/fusion.py`). Latency is measured per retriever.

### Stage E4 — Cross-Encoder Reranking

**File:** `evidence/pipeline.py`, `evidence/reranker.py`

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores every fused candidate against the query
- Drops below `SAH_RELEVANCE_THRESHOLD` (default: −2.0)
- Runs on MPS (Apple Silicon) automatically

### Stage E4 — Evidence Expansion

**File:** `evidence/expander.py`

Packs evidence items into a token-budgeted context string (default: 1800 tokens via tiktoken `cl100k_base`). Each item gets an `[E#]` evidence ID for citation referencing.

### Stage E5 — Evidence Validation

**File:** `evidence/validator.py`

Cross-encoder query–passage scoring across multiple query aspects. Returns:
- `sufficient: bool`
- `coverage: float`
- `max_score: float`
- `missing_aspects: list[str]`

If insufficient and retries remain, the `EvidenceRetrier` queries missing aspects either via structure-guided re-selection (`SAH_RETRY_MODE=structure`) or plain re-query.

### Stage E5 — Grounded Generation

**Files:** `generation/generator.py`, `generation/answerer.py`, `generation/prompt.py`

- Local Ollama (`gemma3:4b`) generates answers with inline `[E1]`-style citation markers
- `INSUFFICIENT_EVIDENCE` sentinel triggers safe abstention
- Claim splitter separates sentences for per-claim citation verification
- `GroundedAnswerer.answer()` drops unsupported claims, surfaces a clean answer

### Stage E5 — Citation Verification

**File:** `generation/verifier.py`

- Cross-encoder NLI (`cross-encoder/nli-deberta-v3-small`) checks each claim against its cited evidence
- Configurable threshold (`SAH_CITATION_NLI_THRESHOLD`, default: 0.5)
- Unsupported claims are silently dropped before the final answer is assembled

---

## Live Trace UI

The React frontend connects to `/chat/stream` (SSE) and renders every pipeline event in real time.

### STAIR Pipeline Visualization

When structural retrieval is active, the UI renders a **5-step progress bar** that animates as each stage completes:

```
⟐ Route Query → 📑 Read ToC → 🎯 LLM Select Sections → 📦 Expand Chunks → ⚡ RRF Fuse
```

Each step shows:
- `pending` (faded)
- `running` (violet glow + spinner)
- `done` (green ✓)
- `warning` (amber ⚠ — keyword fallback path)

### SSE Event Stages

| Stage key | Fired by | Data |
|-----------|----------|------|
| `request` | API on query receive | `session_id`, `query` |
| `query` | After preprocessing | `intent`, `expanded_queries` |
| `pipeline` | SAH vs baseline branch | `mode`, feature flags |
| `routing` | `AdaptiveRetriever._route()` | `retrievers`, `complexity`, `k`, `reason`, `signals` |
| `retriever` | Each retriever start/done | `retriever`, `count`, `latency_ms` |
| `structural_select` | `StructuralRetriever.select_sections()` | `strategy`, `model`, `selected_sections`, `leaf_sections`, `retries`, `reason` |
| `structural_expand` | `StructuralRetriever.retrieve_structural()` | `selected_sections`, `chunk_count`, `matched_chunks`, `strategy`, `retries` |
| `fusion` | After RRF | `fused_count`, `counts` |
| `rerank` | Cross-encoder | `candidates`, `returned`, `model` |
| `evidence` | After expansion | `items`, `token_count`, `sections`, `truncated` |
| `validation` | Sufficiency check | `sufficient`, `coverage`, `max_score`, `missing_aspects` |
| `retry` | Evidence retry loop | `attempt`, `mode`, `new_chunks`, `sufficient_after` |
| `generation` | LLM call | `provider`, `grounded`, `citations` |
| `complete` | End of pipeline | `elapsed_ms`, `mode` |

### Static Trace (after answer)

After generation, the UI shows:
- **Retrieval Mode badge** — color-coded: green (STAIR LLM), amber (keyword fallback), red (no sections)
- **Retrieval Trace panel** — mode, retrievers, complexity, structural strategy, section counts, evidence tokens, routing reason, validation result, retries, generation provider
- **Sources panel** — every evidence chunk with section title, doc name, page, and passage preview
- **Citations** — inline `[E#]` chips in the answer text, expandable to full passage + chunk/section IDs

---

## Project Structure

```
rag-system/
│
├── api/
│   ├── main.py              # FastAPI app — /chat, /chat/stream, /upload, /documents
│   └── trace.py             # Citation & trace serialization helpers
│
├── retrieval/               # SAH retrieval layer
│   ├── types.py             # Chunk dataclass
│   ├── baseline.py          # Weighted RRF over BM25 + dense (E1 wrapper)
│   ├── structural.py        # STAIR-inspired structural retriever (E2) ★
│   ├── adaptive.py          # Parallel multi-retriever orchestrator (E3) ★
│   ├── controller.py        # Rule-based query routing (E3) ★
│   ├── fusion.py            # Reciprocal Rank Fusion (RRF) with dedup
│   └── ollama_client.py     # Minimal Ollama HTTP client
│
├── structure/               # Document structure layer
│   ├── extractor.py         # PDF/DOCX/MD/TXT section tree extraction ★
│   ├── store.py             # SQLite + JSON persistence (StructureStore) ★
│   ├── ingest.py            # Upload → extract → store pipeline
│   ├── chunking.py          # Section-aware chunking
│   └── models.py            # ExtractedDocument, Section, StructureChunk types
│
├── evidence/                # Evidence pipeline (E4/E5)
│   ├── pipeline.py          # Gather: retrieve → rerank → expand → validate → retry ★
│   ├── expander.py          # Token-budget evidence packing
│   ├── reranker.py          # Cross-encoder reranker
│   ├── validator.py         # Sufficiency scoring
│   ├── retry.py             # Structure-guided / plain retry
│   └── types.py             # EvidenceSet, ValidationResult
│
├── generation/              # Grounded generation (E5)
│   ├── generator.py         # GroundedGenerator (Ollama)
│   ├── answerer.py          # Orchestrate: generate → verify → abstain
│   ├── prompt.py            # Citation-first prompt builder
│   ├── claims.py            # Sentence/claim splitter with [E#] extraction
│   ├── verifier.py          # NLI citation verifier
│   └── types.py             # Citation, Claim, GroundedAnswer
│
├── core/                    # Original 23-component baseline
│   ├── query_processing/    # Preprocessor, IntentClassifier, QueryExpander
│   ├── retrieval/           # TFIDFRetriever, VectorDBRetriever, HybridFusion, Reranker
│   ├── generation/          # ContextBuilder, PromptBuilder, LLMEngine, ResponseFormatter
│   └── ingestion/           # Loader, Splitter, Embedder, Indexer, Storage
│
├── utils/
│   ├── progress.py          # emit_progress() — transport-agnostic SSE events ★
│   └── observability.py     # SystemLogger
│
├── ui/                      # React + Vite frontend
│   └── src/
│       ├── App.jsx           # Full UI — chat, live trace, STAIR viz, source explorer ★
│       └── App.css           # Charcoal + amber + violet design system
│
├── sah_config.py            # SAHConfig — all feature flags from environment ★
├── run.py                   # One-command launcher (backend + UI)
├── .env                     # Environment variables (not committed)
└── data/
    ├── sah_rag.db           # SQLite structure store
    └── structure/           # JSON mirrors of section trees
```

★ = added or significantly changed as part of the STAIR / SAH update

---

## Configuration

### `sah_config.py` — `SAHConfig`

All settings are read from environment variables with safe defaults. The baseline E1 path is active when `SAH_ENABLED=false` or all feature flags are off.

| Flag | Default | Description |
|------|---------|-------------|
| `SAH_ENABLED` | `true` | Master switch — disables all SAH stages |
| `SAH_STRUCTURE_ENABLED` | `true` | Extract and store section trees on upload |
| `SAH_STRUCTURAL_RETRIEVAL` | `false` | Enable STAIR-inspired section selection |
| `SAH_ADAPTIVE_CONTROLLER` | `false` | Route by query complexity |
| `SAH_RERANK` | `false` | Cross-encoder reranking |
| `SAH_EVIDENCE_VALIDATION` | `false` | Sufficiency check + retry |
| `SAH_GROUNDED_GENERATION` | `false` | Ollama grounded generation |
| `SAH_CITATION_VERIFICATION` | `false` | NLI per-claim verification |
| `SAH_ABSTAIN_ON_INSUFFICIENT` | `true` | Return abstention message if validation fails |
| `SAH_RETRY_MODE` | `structure` | `structure` or `plain` retry on insufficient evidence |
| `SAH_MAX_RETRIES` | `2` | Maximum evidence retry attempts |
| `SAH_TOP_K` | `5` | Retrieval depth |
| `SAH_STRUCTURAL_TOP_N` | `3` | Max sections to select per query |
| `SAH_CONTEXT_TOKEN_BUDGET` | `1800` | Evidence context token cap |
| `SAH_RELEVANCE_THRESHOLD` | `-2.0` | Cross-encoder score floor |
| `SAH_SUFFICIENCY_THRESHOLD` | `0.35` | Minimum score for evidence sufficiency |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Model for section selection and generation |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `SAH_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.11 | Backend |
| Node.js | ≥ 18 | Frontend |
| Ollama | ≥ 0.3 | Local LLM for STAIR selection + generation |
| `gemma3:4b` model | — | Pulled via Ollama |

### 1. Install Ollama and pull the model

```bash
# macOS
brew install ollama
ollama serve &          # starts the local server on :11434
ollama pull gemma3:4b   # ~3.3 GB download
```

### 2. Clone and set up the Python environment

```bash
git clone <repo-url>
cd rag-system

conda create -n rag_env python=3.11
conda activate rag_env
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env   # or create .env manually
```

Minimum `.env` for STAIR + grounded generation:

```dotenv
# Vector store
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=ragsystem
PINECONE_ENVIRONMENT=us-east-1

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# SAH feature flags
SAH_STRUCTURAL_RETRIEVAL=true
SAH_GROUNDED_GENERATION=true
SAH_ADAPTIVE_CONTROLLER=true    # optional — enables query routing
SAH_RERANK=true                 # optional — cross-encoder reranking
```

### 4. Install frontend dependencies

```bash
cd ui && npm install && cd ..
```

### 5. Start the system

```bash
python run.py
```

This launches:
- **Backend** at `http://localhost:8000` (FastAPI + uvicorn, hot-reload)
- **Frontend** at `http://localhost:5173` (Vite dev server)

Or run separately:

```bash
# Terminal 1 — backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd ui && npm run dev
```

### 6. Upload a document and ask a question

1. Open `http://localhost:5173`
2. Drag a PDF, DOCX, MD, or TXT into the sidebar upload box
3. Wait for the structure extraction confirmation
4. Type a question — the live STAIR trace appears immediately

---

## API Reference

### `POST /chat`
Synchronous chat — returns after full generation.

**Request:**
```json
{ "query": "What is ACSL?", "session_id": "optional-string" }
```

**Response:**
```json
{
  "session_id": "...",
  "response": {
    "answer": "ACSL (Annotated C Specification Language) is... [E1]",
    "citations": [
      {
        "evidence_id": "E1",
        "chunk_id": "...",
        "section_id": "acsl-doc::s0003",
        "section_title": "What is ACSL?",
        "doc": "7-ACSL.pdf",
        "page": 4,
        "role": "primary",
        "score": 0.87,
        "text": "..."
      }
    ],
    "context_chunks": [...],
    "abstained": false,
    "verification": null
  },
  "intent_detected": "factual",
  "expanded_queries": ["What is ACSL?", "ACSL definition"],
  "trace": {
    "mode": "sah",
    "controller": { "retrievers": ["semantic","lexical","structural"], "complexity": "simple", ... },
    "retrieval": {
      "counts": { "semantic": 5, "lexical": 3, "structural": 3 },
      "selected_sections": [{ "section_id": "...", "title": "What is ACSL?", "rank": 1, "chunk_count": 4 }],
      "structural_strategy": "llm",
      "structural_reason": "LLM selected 1 valid leaf section."
    },
    "evidence": { "token_count": 940, "sections": [...], "truncated": false },
    "validation": null,
    "retries": [],
    "generation": { "provider": "ollama:gemma3:4b", "grounded": true }
  }
}
```

### `POST /chat/stream`
Server-Sent Events — streams progress events then a `final` payload.

Each SSE `data:` line is a JSON object:
```json
{ "type": "progress", "stage": "structural_select", "status": "done",
  "message": "STAIR-inspired selector chose 1 section(s) with gemma3:4b.",
  "details": { "strategy": "llm", "selected_sections": [...], "model": "gemma3:4b", "retries": 0 },
  "ts": 1730471560.123 }
```

Final payload: `{ "type": "final", "data": { ...same as /chat... } }`

### `POST /upload`
Upload and index a document (multipart/form-data, field `file`).

Returns:
```json
{
  "status": "success",
  "message": "Successfully uploaded and indexed 7-ACSL.pdf",
  "chunks": 142,
  "filename": "7-ACSL.pdf",
  "structure": {
    "doc_id": "7-acsl-a3f2b1c4",
    "strategy": "pdf_toc",
    "sections": 31,
    "chunks": 142
  }
}
```

### `GET /documents`
List all indexed documents with section counts.

### `GET /documents/{doc_id}/structure`
Return a document's full section tree and chunk-to-section mapping.

### `GET /health`
```json
{ "status": "healthy", "vector_db": "pinecone" }
```

---

## Environment Variables

```dotenv
# ── OpenRouter / OpenAI (query expansion, non-grounded generation) ──
OPEN_ROUTER_API=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

# ── Pinecone ──
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=ragsystem
PINECONE_ENVIRONMENT=us-east-1

# ── CORS ──
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# ── Ollama ──
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT=120

# ── SAH Feature Flags ──
SAH_ENABLED=true
SAH_STRUCTURE_ENABLED=true
SAH_STRUCTURAL_RETRIEVAL=true
SAH_ADAPTIVE_CONTROLLER=false
SAH_RERANK=false
SAH_EVIDENCE_VALIDATION=false
SAH_GROUNDED_GENERATION=true
SAH_CITATION_VERIFICATION=false
SAH_ABSTAIN_ON_INSUFFICIENT=true
SAH_RETRY_MODE=structure
SAH_MAX_RETRIES=2
SAH_TOP_K=5
SAH_STRUCTURAL_TOP_N=3
SAH_CONTEXT_TOKEN_BUDGET=1800
SAH_RELEVANCE_THRESHOLD=-2.0
SAH_SUFFICIENCY_THRESHOLD=0.35
SAH_DB_PATH=data/sah_rag.db
SAH_STRUCTURE_JSON_DIR=data/structure
SAH_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Feature Flags & Experiments

Use `config_for_experiment()` from `sah_config.py` to activate a named experiment:

| Experiment | Flags activated |
|-----------|----------------|
| `E1` | Baseline only (structure disabled) |
| `E2` | + `structural_retrieval` |
| `E3` | + `adaptive_controller` |
| `E4` | + `rerank`, `evidence_expansion` |
| `E5` | + `evidence_validation`, `grounded_generation`, `citation_verification` |

```python
from sah_config import config_for_experiment
config = config_for_experiment("E5")
```

---

## Troubleshooting

### "Ollama selection unavailable" / keyword fallback active

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve &

# Confirm the model is present
ollama list
```

The system auto-recovers within **30 seconds** of Ollama coming back online — no server restart required.

### "Address already in use" on port 8000

```bash
lsof -ti :8000 | xargs kill -9
python run.py
```

### Abstained — evidence was insufficient

This means:
- `SAH_EVIDENCE_VALIDATION=true` and the cross-encoder scored the retrieved passages below `SAH_SUFFICIENCY_THRESHOLD`, **or**
- `SAH_GROUNDED_GENERATION=true` but Ollama is down — generation fails closed and abstains safely

Solutions:
1. Upload more relevant documents
2. Lower `SAH_SUFFICIENCY_THRESHOLD` (e.g., `0.2`) for less strict gating
3. Set `SAH_ABSTAIN_ON_INSUFFICIENT=false` to always attempt generation
4. Check `ollama serve` is running if grounded generation is enabled

### Pinecone SSL / connection errors

Check your `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`. The system falls back to in-memory storage automatically when Pinecone is unreachable, but this does not persist across restarts.

### OpenRouter 402 — Payment Required

OpenRouter credits are exhausted. The query expansion and non-grounded generation path uses OpenRouter. Switch to:
```dotenv
SAH_GROUNDED_GENERATION=true   # use local Ollama instead
```
Or top up your OpenRouter account.

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `fastapi` | 0.111 | REST + SSE API |
| `uvicorn` | 0.30 | ASGI server |
| `sentence-transformers` | 5.7 | Embeddings + cross-encoder reranking |
| `rank-bm25` | 0.2 | BM25 lexical retrieval |
| `PyMuPDF` | ≥1.24 | PDF structure + text extraction |
| `python-docx` | ≥1.1 | DOCX structure extraction |
| `tiktoken` | 0.5 | Token counting for evidence budget |
| `pinecone` | 6.0 | Vector database |
| `requests` | 2.31 | Ollama HTTP client |
| `Ollama` | ≥0.30 (external) | Local LLM — `gemma3:4b` |

---

*Built with ❤️ — SAH-RAG · Structure-Aware Adaptive Hybrid RAG*
