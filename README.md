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
Accepts `.txt` or `.pdf` files via `multipart/form-data` and triggers incremental indexing.

---

## 📊 Evaluation & Observability

The `SystemLogger` component automatically records pipeline metrics to `rag_system.log`. This file tracks:
- End-to-end request latency.
- Sub-pipeline latencies (Query Processing, Retrieval, Fusion, Generation).
- The raw generated responses and context payloads.

This log file can be parsed for offline evaluation frameworks (like RAGAS or TruLens) to compute Answer Relevance and Context Precision.

---
*Built as a showcase for Advanced Agentic Architecture.*
