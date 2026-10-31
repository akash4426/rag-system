# Enterprise RAG System

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenRouter-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)

An enterprise-grade, highly modular Retrieval-Augmented Generation (RAG) system composed of 23 distinct components. This system implements an advanced, production-ready pipeline utilizing Hybrid Search (Dense + Sparse), Reciprocal Rank Fusion (RRF), Cross-Encoder Re-ranking, and dynamic query expansion via the OpenRouter API. Final-answer generation automatically falls back to a local Ollama model when OpenRouter is unconfigured or fails.

It features a sleek, dark-themed "Data Console" built in React, allowing users to effortlessly upload, index, and query documents seamlessly.

---

## 🏗️ Architecture Overview

The system strictly adheres to a modular class-per-component design, logically separated into 4 primary pipelines and an orchestration/UI layer:

### 1. Ingestion Pipeline
- **Document Loader:** Ingests `.txt`, `.pdf`, `.md`, `.markdown`, and `.docx` files.
- **Text Splitter:** Implements recursive character chunking.
- **Embedding Generator:** Generates dense vector representations using `SentenceTransformers`.
- **Storage:** Persists vector data to **Pinecone** (production) or an in-memory cosine store (local fallback).
- **Index Builder:** Manages the ingestion orchestration and builds the offline BM25 sparse index.

### 2. Query Pipeline
- **Query Preprocessor:** Normalizes, cleans, and standardizes incoming queries.
- **Intent Classifier:** Classifies the query (e.g., Factual vs Analytical) to adjust retrieval weights.
- **Query Expander:** Uses the LLM to dynamically generate alternative search phrasing.

### 3. Retrieval Pipeline
- **TF-IDF / BM25 Retriever:** Performs keyword-based sparse search.
- **Vector DB Retriever:** Performs dense semantic search via Pinecone or the in-memory cosine store.
- **Hybrid Fusion Engine:** Merges sparse and dense results using Reciprocal Rank Fusion.
- **Dynamic Weight Controller:** Adjusts BM25/Vector weights based on user intent.
- **Re-ranker:** Uses a `Cross-Encoder` to intelligently sort the fused results.

### 4. Generation Pipeline
- **Context Builder:** Stitches re-ranked text chunks into an optimized context block.
- **Token Manager:** Enforces strict token limits via `tiktoken` to prevent context overflow.
- **Prompt Builder:** Fuses query, chat history, and context into the final LLM prompt.
- **LLM Engine:** Generates the final output using the OpenRouter API (`gpt-4o-mini`), with an automatic local Ollama fallback (`gemma3:4b`) when OpenRouter fails or is unconfigured.
- **Response Formatter:** Extracts and structures the system citations.
- **Memory Module:** Maintains a sliding-window session history.

### 5. API & UI Layer
- **System Logger:** Captures offline generation metrics and processing latency.
- **FastAPI Layer:** Provides a CORS-enabled RESTful API (`/chat`, `/upload`, `/documents`, `/documents/{doc_id}/structure`).
- **React App:** A sleek Vite-powered frontend featuring an interactive Data Console sidebar for document uploading, retrieval traces, clickable citations, and a section-level source explorer.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm
- Ollama with `gemma3:4b` for local generation, structural section selection, and the OpenRouter fallback:
  ```bash
  ollama pull gemma3:4b
  ```

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
   Copy the example environment file and configure the providers:
   ```bash
   cp .env.example .env
   ```
   
   **OpenRouter (primary generator):**
   ```env
   OPEN_ROUTER_API=your_openrouter_key
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
   If OpenRouter is unavailable or out of credits, final generation automatically falls back to Ollama. Set `OLLAMA_FALLBACK_ENABLED=false` to disable this.

   **Local Ollama (fallback and SAH generation/selection):**
   ```env
   OLLAMA_FALLBACK_ENABLED=true
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gemma3:4b
   OLLAMA_TIMEOUT=120
   ```

   **Production with Pinecone (optional):**
   - Get your Pinecone API key from [pinecone.io](https://www.pinecone.io/)
   - Add to `.env`:
     ```env
     PINECONE_API_KEY=your_api_key_here
     PINECONE_ENVIRONMENT=us-east-1
     PINECONE_INDEX_NAME=rag-system
     ```
   - Without Pinecone, the system uses the in-memory cosine store.
   
   See [PINECONE_SETUP.md](./PINECONE_SETUP.md) for detailed setup instructions.

### Running the System

**Method 1: Docker**
You can build and run both the backend and the prebuilt React UI with Docker Compose:
```bash
docker compose up --build
```
The UI and API are served together at `http://localhost:8000`. Add `OPEN_ROUTER_API`, `OPENROUTER_MODEL`, `OLLAMA_*`, and any Pinecone variables to the `environment` block in `docker-compose.yml`; the container does not read your host `.env` automatically. Ollama runs on the host by default, so Docker Desktop users may need `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

**Method 2: Local Development**
Start Ollama only if it is not already running:
```bash
ollama serve
```
> On macOS the Ollama menu-bar app normally starts the server automatically. If `ollama serve` reports `address already in use`, the server is already running; skip this command.

In a second terminal, launch both the FastAPI backend and the Vite UI:
```bash
python run.py
```
The React UI opens at `http://localhost:5173` and talks to the FastAPI backend at `http://localhost:8000`.

To force the structural retriever for every query and keep final generation local:
```bash
export SAH_STRUCTURAL_RETRIEVAL=true
export SAH_ADAPTIVE_CONTROLLER=false
export SAH_GROUNDED_GENERATION=true
export OLLAMA_MODEL=gemma3:4b
python run.py
```

These are backend environment variables, not UI toggles. Restart `run.py` after changing them.

---

## 📂 Document Management

You do not need to run offline indexing scripts. Launch the application and use the **Data Console** sidebar on the left to upload PDF, DOCX, Markdown, or text files.

The system incrementally indexes the document into the dense vector store, BM25 sparse index, and SAH-RAG section store. The upload response reports the structure strategy and section/chunk counts, and the **Source Explorer** lists uploaded documents with their extracted section trees.

If a PDF has bookmarks, those become the section hierarchy. Otherwise the extractor falls back to heading detection, then fixed page windows.

---
## 🔌 API Reference

The FastAPI backend exposes these endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Runtime health and vector-store backend |
| `/chat` | POST | Query the full RAG pipeline and return an answer, citations, and trace |
| `/upload` | POST | Upload and index a document; also builds the SAH section store |
| `/documents` | GET | List structured documents and section counts |
| `/documents/{doc_id}/structure` | GET | Return a document's section tree and chunk mapping |

### 1. Chat Completion (`POST /chat`)

```json
// Request
{
  "query": "What is the order of volatility?",
  "session_id": "optional-uuid-for-memory"
}

// Response (abridged)
{
  "session_id": "uuid",
  "response": {
    "answer": "The order of volatility is ... [E1]",
    "citations": [
      {
        "evidence_id": "E1",
        "chunk_id": "chunk-123",
        "section_id": "doc::s0004",
        "section_title": "Order of Volatility",
        "doc": "notes.pdf",
        "page": 12,
        "text": "..."
      }
    ],
    "context_chunks": [],
    "abstained": false,
    "verification": null
  },
  "intent_detected": "factual",
  "expanded_queries": ["order of volatility", "volatile data order"],
  "trace": {
    "mode": "baseline",
    "retrieval": {"retrievers": ["bm25", "vector_db"], "candidates": 20, "returned": 3},
    "generation": {"provider": "ollama:gemma3:4b", "grounded": false}
  }
}
```

The `trace.generation.provider` field reports the provider that actually answered. It is `openrouter` when the primary succeeds and `ollama:gemma3:4b` when the fallback is used.

### 2. Document Upload (`POST /upload`)

Accepts `.txt`, `.pdf`, `.md`, `.markdown`, and `.docx` files via `multipart/form-data`. The response includes the legacy chunk count and, when structure extraction succeeds, a `structure` object:

```json
{
  "status": "success",
  "message": "Successfully uploaded and indexed notes.pdf",
  "chunks": 29,
  "filename": "notes.pdf",
  "structure": {
    "doc_id": "notes-1a2b3c4d",
    "strategy": "pdf_bookmarks",
    "sections": 32,
    "chunks": 29
  }
}
```

For the extended structured `/chat` response with retrieval traces, see the **SAH-RAG** section below.

---

## 🦙 Local Ollama Generation and Fallback

OpenRouter remains the primary generator for the legacy path. When it is unconfigured, returns an error such as `402 Insufficient credits`, times out, or returns an empty response, `LLMEngine` retries the same prompt against local Ollama.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_FALLBACK_ENABLED` | `true` | Enable the automatic OpenRouter → Ollama fallback |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma3:4b` | Local generation and structural-selection model |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |

The API trace reports which provider answered:

```json
"generation": {
  "provider": "ollama:gemma3:4b",
  "grounded": false
}
```

Query expansion still uses OpenRouter when configured. If that call fails, expansion degrades to the original query; final-answer generation is the stage protected by the Ollama fallback.

### Fully local mode

To skip OpenRouter entirely, either enable grounded SAH generation or leave `OPEN_ROUTER_API` unset:

```bash
export SAH_GROUNDED_GENERATION=true
export OLLAMA_MODEL=gemma3:4b
python run.py
```

Grounded generation calls Ollama directly and does not attempt OpenRouter.

### Ollama troubleshooting

- `bind: address already in use` on port `11434` means Ollama is already running. Do not start a second server.
- Check the server and model:
  ```bash
  curl -s http://localhost:11434/api/tags
  ollama list
  ```
- Pull the model if it is missing:
  ```bash
  ollama pull gemma3:4b
  ```
- On macOS, the Ollama menu-bar app and `ollama serve` start the same server. Use one or the other.

---

## 📊 Evaluation & Observability

The `SystemLogger` component automatically records pipeline metrics to `rag_system.log`. This file tracks:
- End-to-end request latency.
- Sub-pipeline latencies (Query Processing, Retrieval, Fusion, Generation).
- The raw generated responses and context payloads.

This log file can be parsed for offline evaluation frameworks (like RAGAS or TruLens) to compute Answer Relevance and Context Precision.

## ✅ Tests

```bash
python -m pytest -q
```

The suite covers structure extraction and mapping, structural selection and fallback, adaptive routing and fusion, evidence expansion, validation and retry modes, grounded generation and citation verification, trace serialization, evaluation metrics, and the OpenRouter → Ollama fallback.

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
| `OLLAMA_FALLBACK_ENABLED` | `true` | Retry final generation on local Ollama when OpenRouter fails |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local generation/selector endpoint |
| `OLLAMA_MODEL` | `gemma3:4b` | Local generation/selector model |
| `OLLAMA_TIMEOUT` | `120` | Local Ollama request timeout in seconds |

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
ollama pull gemma3:4b
```

See [Local Ollama Generation and Fallback](#-local-ollama-generation-and-fallback) for server startup and troubleshooting.

The legacy OpenRouter generation path now retries the same prompt against this local model when OpenRouter is unconfigured, returns an error (for example, insufficient credits), or returns an empty response. Set `OLLAMA_FALLBACK_ENABLED=false` to disable that fallback. The response trace reports the provider that actually answered (`openrouter` or `ollama:gemma3:4b`).

### Testing structural retrieval from the UI

The SAH flags are backend process settings, not switches inside the React app.

1. Start Ollama (skip if it is already running) and confirm the model:
   ```bash
   ollama list
   ```
2. Start the app with structural retrieval and local grounded generation:
   ```bash
   export SAH_STRUCTURAL_RETRIEVAL=true
   export SAH_ADAPTIVE_CONTROLLER=false
   export SAH_GROUNDED_GENERATION=true
   export OLLAMA_MODEL=gemma3:4b
   python run.py
   ```
   `SAH_ADAPTIVE_CONTROLLER=false` makes the structural route run for every query. With the controller enabled, simple factual queries may use only semantic and lexical retrieval.
3. In the left **Data Console**, upload a PDF, DOCX, Markdown, or text file. Wait for `Successfully uploaded and indexed ...`. The **Source Explorer** should list the document and show a section count such as `32§`.
4. Expand the document in **Source Explorer** to confirm the extracted section tree.
5. Ask a section-oriented question, for example:
   > Which section discusses indirect triples?
6. Open **Retrieval Trace** under the answer and verify:
   - **Mode:** `sah`
   - **Retrievers:** `semantic + lexical + structural`
   - **Structural Strategy:** `llm`
   - **Retrieval Counts:** includes `structural: N` with `N > 0`
   - **Selected Sections:** up to `SAH_STRUCTURAL_TOP_N` section titles
   - **Generation:** `ollama:gemma3:4b` when grounded generation is enabled
7. Click **Sources** to inspect retrieved passages and click an inline `[E#]` citation to see its document, page, section, and chunk ID.

Each answer also shows a retrieval-mode badge above the metadata buttons. Click the badge to open the trace.

| Badge | Meaning |
|---|---|
| Green · **Structural retrieval · STAIR-inspired** | Structural chunks contributed to the retrieved evidence |
| Amber · **Structural retrieval · keyword fallback** | The Ollama selector failed; deterministic section matching supplied structural chunks |
| Amber · **Structural route ran · no chunks returned** | Sections were selected but produced no usable chunks |
| Red · **Structural route failed · no sections** | Structure extraction returned no sections for the uploaded documents |
| Grey · **Baseline hybrid retrieval** | The request used the legacy BM25 + dense path |
| Grey · **Adaptive hybrid retrieval** | SAH mode was active, but the structural route did not contribute |

`Structural Strategy` meanings:

| Value | Meaning |
|---|---|
| `llm` | Ollama selected valid section IDs; this is the STAIR-inspired route |
| `keyword_fallback` | Ollama was unavailable or returned malformed JSON; deterministic keyword matching selected sections |
| `no_sections` | No structure was stored for the requested documents; re-upload or check `GET /documents` |

If structural retrieval does not appear in the trace, confirm the backend was restarted after exporting the flags.

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
    "controller": {
      "retrievers": ["semantic", "lexical", "structural"],
      "complexity": "fixed"
    },
    "retrieval": {
      "counts": {"semantic": 5, "lexical": 5, "structural": 3},
      "selected_sections": [
        {"section_id": "doc::s0011", "title": "Three-Address Code", "rank": 1}
      ],
      "structural_strategy": "llm"
    },
    "validation": {"sufficient": true, "coverage": 1.0},
    "retries": [],
    "evidence": {"token_count": 823, "items": []},
    "generation": {
      "provider": "ollama:gemma3:4b",
      "grounded": true,
      "abstained": false
    }
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
- The legacy OpenRouter generation path is retained with an automatic Ollama fallback. Grounded SAH generation requires Ollama.
- The Ollama fallback improves availability, not answer quality; `gemma3:4b` may be slower and less accurate than the configured OpenRouter model.
- Query expansion still depends on OpenRouter when configured. If it fails, the system proceeds with the original query rather than spending another local generation call.

---
*Built as a showcase for Advanced Agentic Architecture.*
