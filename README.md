# Enterprise RAG System

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenRouter-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)

An enterprise-grade, highly modular Retrieval-Augmented Generation (RAG) system composed of 23 distinct components. This system implements an advanced, production-ready pipeline utilizing Hybrid Search (Dense + Sparse), Reciprocal Rank Fusion (RRF), Cross-Encoder Re-ranking, and dynamic query expansion via the OpenRouter API.

It features a sleek, dark-themed "Data Console" built in React, allowing users to effortlessly upload, index, and query documents seamlessly.

---

## 🏗️ Architecture Overview

The system strictly adheres to a modular class-per-component design, logically separated into 4 primary pipelines and an orchestration/UI layer:

### 1. Ingestion Pipeline
- **Document Loader:** Ingests raw `.txt` and `.pdf` files.
- **Text Splitter:** Implements recursive character chunking.
- **Embedding Generator:** Generates dense vector representations using `SentenceTransformers`.
- **Storage:** Persists vector data to **Pinecone** (production) or ChromaDB (fallback).
- **Index Builder:** Manages the ingestion orchestration and builds the offline BM25 sparse index.

### 2. Query Pipeline
- **Query Preprocessor:** Normalizes, cleans, and standardizes incoming queries.
- **Intent Classifier:** Classifies the query (e.g., Factual vs Analytical) to adjust retrieval weights.
- **Query Expander:** Uses the LLM to dynamically generate alternative search phrasing.

### 3. Retrieval Pipeline
- **TF-IDF / BM25 Retriever:** Performs keyword-based sparse search.
- **Vector DB Retriever:** Performs dense semantic search via Pinecone or ChromaDB.
- **Hybrid Fusion Engine:** Merges sparse and dense results using Reciprocal Rank Fusion.
- **Dynamic Weight Controller:** Adjusts BM25/Vector weights based on user intent.
- **Re-ranker:** Uses a `Cross-Encoder` to intelligently sort the fused results.

### 4. Generation Pipeline
- **Context Builder:** Stitches re-ranked text chunks into an optimized context block.
- **Token Manager:** Enforces strict token limits via `tiktoken` to prevent context overflow.
- **Prompt Builder:** Fuses query, chat history, and context into the final LLM prompt.
- **LLM Engine:** Generates the final output using the OpenRouter API (`gpt-4o-mini`).
- **Response Formatter:** Extracts and structures the system citations.
- **Memory Module:** Maintains a sliding-window session history.

### 5. API & UI Layer
- **System Logger:** Captures offline generation metrics and processing latency.
- **FastAPI Layer:** Provides a CORS-enabled, RESTful API (`/chat`, `/upload`).
- **React App:** A sleek Vite-powered frontend featuring an interactive Data Console sidebar for document uploading and real-time processing metadata visibility.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd rag-system
   ```

2. **Install Python Backend Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install React Frontend Dependencies:**
   ```bash
   cd ui
   npm install
   cd ..
   ```

4. **Environment Variables:**
   Copy the example environment file and insert your API keys:
   ```bash
   cp .env.example .env
   ```
   
   **For Production with Pinecone:**
   - Get your Pinecone API key from [pinecone.io](https://www.pinecone.io/)
   - Add to `.env`:
     ```env
     PINECONE_API_KEY=your_api_key_here
     PINECONE_ENVIRONMENT=us-east-1
     OPENAI_API_KEY=your_openai_key_here
     ```
   
   **For Local Development (Fallback):**
   - Leave Pinecone variables empty to use in-memory storage
   - Only OPENAI_API_KEY is required
   
   See [PINECONE_SETUP.md](./PINECONE_SETUP.md) for detailed setup instructions.

### Running the System

**Method 1: Docker (Recommended for Production)**
You can instantly spin up both the backend and frontend, fully containerized, using Docker Compose:
```bash
docker compose up --build
```
The React UI will be available at `http://localhost:3000` and the API at `http://localhost:8000`. Your Vector Database is safely persisted in a Docker Volume.

**Method 2: Local Development**
You can boot both the FastAPI backend and the React development server concurrently using the provided start script:
```bash
python run.py
```
The React UI will automatically open in your default browser at `http://localhost:5173`, communicating with the FastAPI backend on `http://localhost:8000`.

---

## 📂 Document Management

You do not need to run offline indexing scripts! Simply launch the application and use the **Data Console** sidebar on the left to upload `.txt` or `.pdf` files. 

The system will incrementally process the document through all 23 components, updating both the VectorDB and BM25 Sparse Index in real-time.

---
## 🔌 API Reference

The FastAPI backend exposes two primary endpoints designed for headless integration:

### 1. Chat Completion (`POST /chat`)
Processes a query through the full 23-component RAG pipeline.
```json
// Request
{
  "query": "What is the order of volatility?",
  "session_id": "optional-uuid-for-memory"
}

// Response
{
  "session_id": "uuid",
  "response": {
    "answer": "The order of volatility is...",
    "citations": ["bm25, vector_db"],
    "context_chunks": [
      {
        "source": "bm25",
        "text": "...extracted text chunk..."
      }
    ]
  },
  "intent_detected": "analytical",
  "expanded_queries": ["explain volatility order", "define order of volatility"]
}
```

### 2. Document Upload (`POST /upload`)
Accepts `.txt`, `.pdf`, `.md`, and `.docx` files via `multipart/form-data` and triggers incremental indexing.

For the extended, structured `/chat` response with retrieval traces, see the **SAH-RAG** section below.

---

## 📊 Evaluation & Observability

The `SystemLogger` component automatically records pipeline metrics to `rag_system.log`. This file tracks:
- End-to-end request latency.
- Sub-pipeline latencies (Query Processing, Retrieval, Fusion, Generation).
- The raw generated responses and context payloads.

This log file can be parsed for offline evaluation frameworks (like RAGAS or TruLens) to compute Answer Relevance and Context Precision.

---

## SAH-RAG

**SAH-RAG** (Structure-Aware Adaptive Hybrid RAG) is the experimental extension built on top of the original hybrid RAG system. The original BM25, dense-vector, fusion, and generation paths are preserved as the baseline. Every SAH stage is independently switchable, so E1-E5 can be run and ablated without changing code.

> The structural retriever is **STAIR-inspired**: it uses an LLM to select sections from a document's table of contents, then expands those sections into chunks. It is not the original STAIR paper or its implementation.
>
> The dense baseline in this repository is Pinecone or the in-memory cosine store. No FAISS layer was added.

### Architecture

```text
Question
  -> legacy query preprocessing
  -> rule-based query controller
       -> semantic (dense)  +  lexical (BM25)  +  structural (STAIR-inspired)
  -> weighted reciprocal-rank fusion
  -> cross-encoder reranking                       (optional)
  -> evidence expansion: parent summary, neighbours, siblings, budgeted
  -> cross-encoder sufficiency validation          (never the generator)
       -> structure-guided retry (max 2)
       -> or plain re-query for the ablation
  -> grounded answer with [E#] citations           (local Ollama)
  -> claim splitting + cross-encoder/NLI citation verification
       -> unsupported claims dropped
       -> abstain when evidence is insufficient
  -> API trace, clickable citations, source explorer
```

### Packages

| Package | Responsibility |
|---|---|
| `structure/` | PDF bookmark/heading/fixed extraction, DOCX/Markdown/TXT sections, SQLite + JSON persistence, chunk-to-section mapping |
| `retrieval/` | Typed baseline wrapper, STAIR-inspired structural selector, rule-based controller, parallel adaptive retrieval, weighted RRF |
| `evidence/` | Parent/neighbour/sibling expansion, cross-encoder reranking, sufficiency validation, structure-guided/plain retry |
| `generation/` | Citation-first prompts, grounded generation, rule-based claim splitting, cross-encoder or NLI claim verification, abstention |
| `eval/` | Retrieval and end-to-end answer evaluation, CSVs, and comparison tables |

The legacy package still owns the original implementation: `core/query_processing/`, `core/retrieval/`, and `core/generation/`. `retrieval/baseline.py` wraps the existing BM25 and dense stores rather than replacing them.

### Experiment definitions

| Experiment | Retrieval / evidence stage | Generation and validation |
|---|---|---|
| **E1** | Legacy BM25 + dense + weighted RRF | Grounded generator (for comparable answer metrics) |
| **E2** | STAIR-inspired structural retrieval | Grounded generator |
| **E3** | Adaptive controller + semantic/lexical/structural RRF | Grounded generator |
| **E4** | E3 + cross-encoder rerank + budgeted context expansion | Grounded generator |
| **E5** | E4 + cross-encoder sufficiency validation + structure-guided retry | Grounded generation + claim-level citation verification + abstention |
| **E5-plain** | E5 with plain re-query retries | Same as E5; ablation only changes retry mode |

`eval/run_eval.py` applies the experiment flags with `config_for_experiment`, so retrieval-only E1-E5 results and answer-level E1-E5 results are reproducible from the same code path.

### Feature flags

All flags are read from the environment by `sah_config.SAHConfig`. The defaults reproduce the baseline path.

| Variable | Default | Effect |
|---|---|---|
| `SAH_ENABLED` | `true` | Master switch for the SAH layer |
| `SAH_STRUCTURE_ENABLED` | `true` | Extract and store document structure during upload |
| `SAH_STRUCTURAL_RETRIEVAL` | `false` | Enable the STAIR-inspired section selector |
| `SAH_ADAPTIVE_CONTROLLER` | `false` | Enable rule-based semantic/lexical/structural routing |
| `SAH_RERANK` | `false` | Cross-encoder reranking of fused candidates |
| `SAH_EVIDENCE_EXPANSION` | `false` | Add parent, neighbour, and sibling context within a token budget |
| `SAH_EVIDENCE_VALIDATION` | `false` | Cross-encoder sufficiency check and retry loop |
| `SAH_GROUNDED_GENERATION` | `false` | Citation-first Ollama generation with abstention |
| `SAH_CITATION_VERIFICATION` | `false` | Verify and drop unsupported claims |
| `SAH_ABSTAIN_ON_INSUFFICIENT` | `true` | Skip generation when validation marks evidence insufficient |
| `SAH_RETRY_MODE` | `structure` | `structure` or `plain` for the retry ablation |
| `SAH_MAX_RETRIES` | `2` | Maximum retry attempts |
| `SAH_TOP_K` | `5` | Retrieval depth used by the API |
| `SAH_STRUCTURAL_TOP_N` | `3` | Maximum sections selected by the structural retriever |
| `SAH_CONTEXT_TOKEN_BUDGET` | `1800` | Evidence-expansion budget |
| `SAH_RELEVANCE_THRESHOLD` | `-2.0` | Cross-encoder validation threshold |
| `SAH_CITATION_THRESHOLD` | `0.0` | Cross-encoder claim-support threshold |
| `SAH_CITATION_VERIFIER` | `cross-encoder` | `cross-encoder` or `nli` |
| `SAH_CITATION_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | Optional NLI verifier |
| `SAH_CITATION_NLI_THRESHOLD` | `0.5` | NLI entailment threshold |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local generation/selector endpoint |
| `OLLAMA_MODEL` | `gemma3:4b` | Local generation/selector model |

To exercise E5 through the API, enable the corresponding flags before starting the backend. The defaults keep `/chat` on the legacy baseline path:

```bash
export SAH_ADAPTIVE_CONTROLLER=true
export SAH_RERANK=true
export SAH_EVIDENCE_EXPANSION=true
export SAH_EVIDENCE_VALIDATION=true
export SAH_GROUNDED_GENERATION=true
export SAH_CITATION_VERIFICATION=true
```

The local generator requires Ollama and a pulled model:

```bash
ollama serve
ollama pull gemma3:4b
```

### Running experiments

Retrieval-only E1-E5:

```bash
python eval/run_eval.py --experiment E1 --ingest --k 5
python eval/run_eval.py --experiment E2 --ingest --k 5
python eval/run_eval.py --experiment E3 --ingest --k 5
python eval/run_eval.py --experiment E4 --ingest --k 5
python eval/run_eval.py --experiment E5 --retry-mode structure --ingest --k 5
python eval/run_eval.py --experiment E5 --retry-mode plain --ingest --k 5
```

End-to-end answer evaluation, including the E5 plain ablation and comparison table:

```bash
python eval/run_eval.py --experiment all --answer-eval --ingest --k 5
```

Outputs:

| File | Contents |
|---|---|
| `eval/results/E*_per_question.csv` | Per-question retrieval metrics and traces |
| `eval/results/E*_summary.csv` | Retrieval summary |
| `eval/results/E*_answer_summary.csv` | Answer, faithfulness, citation, abstention, latency, token summary |
| `eval/results/comparison.csv` | Machine-readable comparison across E1-E5 and E5-plain |
| `eval/results/comparison.md` | Markdown comparison table |

### Answer metrics

| Metric | Definition |
|---|---|
| `recall@k`, `mrr`, `ndcg@k`, `precision@k` | Gold-section retrieval quality |
| `answer_f1` | Token F1 between the answer and `gold_answer`, with citation markers removed |
| `answer_similarity` | Cosine similarity between MiniLM embeddings of the answer and gold answer |
| `faithfulness` | Supported claims / total generated claims |
| `citation_accuracy` | Supported claims / claims that carry citations |
| `citation_precision` | Resolving citation references / all citation references |
| `citation_coverage` | Claims with at least one citation / total claims |
| `decision_accuracy` | Correct answer-vs-abstain decision over all questions |
| `unanswerable_abstention_rate` | Abstention rate on unanswerable questions |
| `false_abstention_rate` | Abstention rate on answerable questions |
| `retrieval_ms`, `generation_ms`, `answer_tokens` | Cost and token accounting |
| `retry_count` | Mean retries for E5 and the E5-plain ablation |

### API trace and source explorer

`POST /chat` preserves the legacy response fields (`response.answer`, `session_id`, `intent_detected`, `expanded_queries`) and adds:

```json
{
  "response": {
    "answer": "... [E1] ...",
    "citations": [
      {
        "evidence_id": "E1",
        "chunk_id": "...",
        "section_id": "...",
        "section_title": "Slide 11: Three-Address Code",
        "doc": "18. INTERMEDIATE CODE GENERATION.pdf",
        "page": 11,
        "role": "primary",
        "text": "..."
      }
    ],
    "context_chunks": [],
    "abstained": false,
    "verification": {
      "claims_total": 3,
      "claims_supported": 3,
      "citation_accuracy": 1.0
    }
  },
  "trace": {
    "mode": "sah",
    "controller": {"retrievers": ["semantic", "lexical"], "complexity": "simple"},
    "retrieval": {"counts": {}, "selected_sections": []},
    "validation": {"sufficient": true, "coverage": 1.0},
    "retries": [],
    "evidence": {"token_count": 823, "items": []}
  }
}
```

The React UI renders inline `[E#]` markers as clickable citation chips, shows the cited passage, exposes a collapsible retrieval trace, and lists documents/sections in the source explorer (`GET /documents` and `GET /documents/{doc_id}/structure`).

### Results

Run on the five-question sample set, one 32-page PDF, with the local `gemma3:4b` generator:

| Experiment | R@5 | P@5 | MRR | nDCG@5 | Answer F1 | Answer sim | Faithful | Citation acc | Citation cov | Decision acc | Unans. abstain | False abstain | Retries | Total ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0.800 | 0.400 | 0.800 | 0.764 | 0.277 | 0.771 | 1.000 | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.00 | 7583 |
| E2 | 0.800 | 0.200 | 0.700 | 0.726 | 0.266 | 0.744 | 0.938 | 0.938 | 1.000 | 0.800 | 0.000 | 0.000 | 0.00 | 11409 |
| E3 | 0.800 | 0.360 | 0.700 | 0.725 | 0.267 | 0.781 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.00 | 10998 |
| E4 | 0.800 | 0.320 | 0.700 | 0.717 | 0.305 | 0.781 | 1.000 | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.00 | 13506 |
| E5-structure | 0.800 | 0.320 | 0.700 | 0.717 | 0.294 | 0.775 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 0.000 | 0.40 | 11472 |
| E5-plain | 0.800 | 0.320 | 0.700 | 0.717 | 0.268 | 0.785 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 0.000 | 0.40 | 10635 |

Key observations:

- Retrieval is flat across variants at Recall@5 = 0.80. E5's validation/retry does not change retrieval for answerable questions because no retry is triggered; the only retried query is the unanswerable one.
- Cross-encoder validation is what makes abstention reliable in this sample: E5 abstains on the unanswerable question every time (decision accuracy 1.0). E1, E2, and E4 answered it from nearby evidence, and E3 abstained by chance rather than by an explicit sufficiency gate.
- E5-structure retry improves answer F1 over the plain re-query ablation (0.294 vs 0.268) and keeps citation support at 0.938, but costs more retrieval latency (3.08 s vs 1.88 s) and total latency (11.5 s vs 10.6 s). The retry count is identical (0.4) because both modes retry only the unanswerable query.
- Citation coverage and citation precision are 1.0 for E5: every retained claim carries a citation and every citation resolves to retrieved evidence. One claim in the quadruples answer was dropped as unsupported, which is why faithfulness/citation accuracy are 0.938 rather than 1.0.
- Answer F1 is modest (0.27-0.31) because the gold answers are short and the local model is verbose; embedding similarity is higher (0.74-0.78). These metrics are reported together rather than treated as interchangeable.
- E1's perfect faithfulness/citation score on the unanswerable question illustrates why answer-level grounding alone is not enough: a claim can be supported by an irrelevant passage. Sufficiency validation is a separate decision.

### Limitations

- The evaluation set contains five questions over one 32-page PDF. Results are illustrative, not statistically significant.
- Answer F1 is sensitive to answer length and paraphrasing; embedding similarity is included as a complement, not a replacement for human judgment.
- The citation verifier uses a cross-encoder relevance score by default. This is not formal entailment; the optional NLI verifier is available behind `SAH_CITATION_VERIFIER=nli`.
- Validation thresholds (`relevance_threshold`, coverage `0.5`, citation threshold) are fixed heuristics and were not tuned on a separate development set.
- Structural selection depends on a local Ollama model. Malformed JSON is retried and falls back to keyword selection; valid section IDs do not guarantee relevance.
- Structure extraction is limited to the implemented strategies: PDF bookmarks, PDF headings, fixed PDF pagination, DOCX headings, and Markdown/TXT headings.
- Retries are capped at two and reuse the same evidence store; a genuinely unanswerable question can still consume the full retry budget.
- Latency measurements come from a local Apple-silicon machine using MPS and are not production benchmarks.
- The legacy OpenRouter generation path remains available for the baseline; grounded SAH generation requires Ollama.

---
*Built as a showcase for Advanced Agentic Architecture.*
