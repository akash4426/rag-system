from fastapi import FastAPI, HTTPException, UploadFile, File
import os
from dotenv import load_dotenv

# Load environment variables before importing components
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import queue
import threading
import time
import uuid

# Import our 23 components
from core.query_processing.preprocessor import QueryPreprocessor
from core.query_processing.intent_classifier import IntentClassifier
from core.query_processing.expander import QueryExpander
from core.retrieval.tfidf_retriever import TFIDFRetriever
from core.retrieval.embedding_model import QueryEmbeddingModel
from core.retrieval.vector_db_retriever import VectorDBRetriever
from core.retrieval.hybrid_fusion import HybridFusionEngine
from core.retrieval.weight_controller import DynamicWeightController
from core.retrieval.reranker import Reranker
from core.generation.context_builder import ContextBuilder
from core.generation.token_manager import TokenManager
from core.generation.prompt_builder import PromptBuilder
from core.generation.llm_engine import LLMEngine
from core.generation.response_formatter import ResponseFormatter
from core.generation.memory import MemoryModule
from utils.observability import SystemLogger
from core.ingestion.indexer import IndexBuilder
from core.ingestion.storage import Storage
from sah_config import SAHConfig
from structure.ingest import ingest_document
from structure.store import StructureStore
from evidence.pipeline import EvidencePipeline
from generation.answerer import GroundedAnswerer
from retrieval.adaptive import AdaptiveRetriever
from retrieval.baseline import BaselineRetriever
from retrieval.structural import StructuralRetriever
from utils.progress import ProgressCallback, emit_progress
from api.trace import (
    citation_from_chunk,
    citation_from_legacy,
    evidence_context_chunks,
    legacy_context_chunks,
    legacy_trace,
)

app = FastAPI(title="23-Component RAG API")

# CORS — allow Vercel frontend and local dev
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline Components
logger = SystemLogger()
preprocessor = QueryPreprocessor()
classifier = IntentClassifier()
expander = QueryExpander()
tfidf = TFIDFRetriever()
embedder = QueryEmbeddingModel()
vector_db = VectorDBRetriever()
fusion = HybridFusionEngine()
weights = DynamicWeightController()
reranker = Reranker()
context_builder = ContextBuilder()
token_manager = TokenManager()
prompt_builder = PromptBuilder()
llm = LLMEngine()
formatter = ResponseFormatter()

# Storage backend (Pinecone or fallback)
storage = Storage(use_pinecone=True)

# SAH-RAG structure store (features are individually switchable)
sah_config = SAHConfig.from_env()
structure_store = StructureStore(
    db_path=sah_config.db_path,
    json_dir=sah_config.structure_json_dir,
)

# SAH-RAG pipeline. Every stage is opt-in through SAH_* environment flags;
# with all flags off, /chat keeps using the untouched baseline path.
sah_baseline = BaselineRetriever(
    tfidf=tfidf,
    vector_db=vector_db,
    embedder=embedder,
    storage=storage,
    structure_store=structure_store,
)
sah_adaptive = AdaptiveRetriever(
    sah_baseline,
    StructuralRetriever(structure_store, config=sah_config),
    config=sah_config,
)
sah_evidence = EvidencePipeline(sah_adaptive, structure_store, config=sah_config)
sah_answerer = GroundedAnswerer(sah_config)


def sah_features_active(config: SAHConfig) -> bool:
    """Return True when any SAH stage is enabled for chat."""
    return config.enabled and any(
        (
            config.structural_retrieval,
            config.adaptive_controller,
            config.rerank,
            config.evidence_expansion,
            config.evidence_validation,
            config.grounded_generation,
            config.citation_verification,
        )
    )

# Simple in-memory session store
sessions = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

@app.on_event("startup")
async def startup_event():
    """Log system status on startup."""
    print("\n" + "="*60)
    print("🚀 RAG System Starting Up")
    print("="*60)
    if storage.use_pinecone:
        print("✓ Vector DB: Pinecone (Production)")
    else:
        print("✓ Vector DB: In-Memory (Development)")
    print("✓ LLM: OpenAI")
    print("✓ Embeddings: Sentence-Transformers")
    stats = structure_store.stats()
    print(f"✓ Structure store: {stats['documents']} docs, {stats['sections']} sections, {stats['chunks']} chunks")
    print("="*60 + "\n")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_db": "pinecone" if storage.use_pinecone else "memory"
    }

def _run_chat(request: QueryRequest, on_event: ProgressCallback | None = None):
    """Run the full chat pipeline and optionally publish progress events."""
    start_time = time.time()

    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = MemoryModule()

    memory = sessions[session_id]
    raw_query = request.query
    logger.log_query(session_id, raw_query)
    emit_progress(
        on_event, "request", "running", "Query received.",
        session_id=session_id, query=raw_query,
    )

    # 1. Query Processing
    t0 = time.time()
    clean_query = preprocessor.preprocess(raw_query)
    intent = classifier.classify(clean_query)
    expanded_queries = expander.expand(clean_query)
    history = memory.get_history_string()
    logger.log_latency("Query Processing", t0)
    emit_progress(
        on_event, "query", "done",
        f"Query processed as {intent}.",
        intent=intent, expanded_queries=expanded_queries,
    )

    abstained = False
    verification = None

    if sah_features_active(sah_config):
        # 2-4. SAH-RAG evidence pipeline + grounded generation.
        emit_progress(
            on_event, "pipeline", "done",
            "SAH-RAG pipeline active; structural retrieval is enabled.",
            mode="sah",
            structural_retrieval=sah_config.structural_retrieval,
            adaptive_controller=sah_config.adaptive_controller,
            rerank=sah_config.rerank,
            evidence_validation=sah_config.evidence_validation,
            grounded_generation=sah_config.grounded_generation,
        )
        t0 = time.time()
        evidence_result = sah_evidence.gather(
            clean_query, k=sah_config.top_k, on_event=on_event
        )
        logger.log_latency("SAH Retrieval & Evidence", t0)

        t0 = time.time()
        if sah_config.grounded_generation:
            provider = f"ollama:{sah_config.ollama_model}"
            emit_progress(
                on_event, "generation", "running",
                f"Generating a grounded answer with {provider}.",
                provider=provider, grounded=True,
                citations=len(evidence_result.evidence.items),
            )
            grounded = sah_answerer.answer(
                clean_query,
                evidence_result.evidence,
                evidence_result.validation,
            )
            answer = grounded.answer
            citations = [citation.to_dict() for citation in grounded.citations]
            abstained = grounded.abstained
            verification = grounded.verification.to_dict() if grounded.verification else None
            generated_text = grounded.raw_response
            grounded_flag = True
        else:
            emit_progress(
                on_event, "generation", "running",
                "Generating from evidence with OpenRouter (Ollama fallback enabled).",
                grounded=False,
            )
            context = evidence_result.evidence.context_text
            prompt = prompt_builder.build(clean_query, context, history)
            prompt = token_manager.truncate_context(prompt, prompt_template_tokens=500)
            answer, provider = llm.generate_with_provider(prompt)
            generated_text = answer
            citations = [
                citation_from_chunk(item, index + 1)
                for index, item in enumerate(evidence_result.evidence.items)
            ]
            grounded_flag = False

        emit_progress(
            on_event, "generation", "done",
            f"Answer generated by {provider}.",
            provider=provider,
            grounded=grounded_flag,
            abstained=abstained,
            citations=len(citations),
            verification=verification,
        )

        memory.add_interaction(clean_query, answer)
        context_chunks = evidence_context_chunks(evidence_result.evidence)
        trace = evidence_result.trace()
        trace["mode"] = "sah"
        trace["query_processing"] = {
            "intent": intent,
            "expanded_queries": expanded_queries,
        }
        trace["generation"] = {
            "provider": provider,
            "grounded": grounded_flag,
            "abstained": abstained,
            "verification": verification,
        }
        logger.log_latency("SAH Generation", t0)
        logger.log_generation(session_id, clean_query, generated_text)
    else:
        # 2. Baseline retrieval: BM25 + dense, weighted fusion, rerank.
        emit_progress(
            on_event, "pipeline", "warning",
            "Baseline retrieval active; SAH structural retrieval is not enabled.",
            mode="baseline",
        )
        t0 = time.time()
        sparse_weight, dense_weight = weights.get_weights(intent)

        emit_progress(
            on_event, "retriever", "running",
            "BM25 lexical retrieval started.", retriever="bm25",
        )
        all_sparse_results = []
        all_dense_results = []

        for q in expanded_queries:
            sparse_res = tfidf.retrieve(q)
            all_sparse_results.extend(sparse_res)

            q_emb = embedder.embed_query(q)
            dense_res = vector_db.retrieve(q_emb)
            all_dense_results.extend(dense_res)

        emit_progress(
            on_event, "retriever", "done",
            f"BM25 returned {len(all_sparse_results)} result(s).",
            retriever="bm25", count=len(all_sparse_results),
        )
        emit_progress(
            on_event, "retriever", "done",
            f"Dense retrieval returned {len(all_dense_results)} result(s).",
            retriever="vector_db", count=len(all_dense_results),
        )

        # 3. Fusion & re-ranking
        fused_results = fusion.fuse(
            all_sparse_results, all_dense_results, sparse_weight, dense_weight
        )
        emit_progress(
            on_event, "fusion", "done",
            f"Weighted fusion produced {len(fused_results)} candidate(s).",
            fused_count=len(fused_results),
            weights={"sparse": sparse_weight, "dense": dense_weight},
        )
        reranked_results = reranker.rerank(clean_query, fused_results, top_k=3)
        emit_progress(
            on_event, "rerank", "done",
            f"Cross-encoder kept {len(reranked_results)} evidence chunk(s).",
            returned=len(reranked_results),
        )
        logger.log_latency("Retrieval & Fusion", t0)

        # 4. Generation
        t0 = time.time()
        emit_progress(
            on_event, "generation", "running",
            "Generating with OpenRouter (Ollama fallback enabled).",
            grounded=False,
        )
        context = context_builder.build_context(reranked_results)
        prompt = prompt_builder.build(clean_query, context, history)
        prompt = token_manager.truncate_context(prompt, prompt_template_tokens=500)

        raw_response, generation_provider = llm.generate_with_provider(prompt)
        memory.add_interaction(clean_query, raw_response)
        logger.log_latency("Generation", t0)
        logger.log_generation(session_id, prompt, raw_response)

        answer = raw_response
        citations = [
            citation_from_legacy(item, index + 1)
            for index, item in enumerate(reranked_results)
        ]
        context_chunks = legacy_context_chunks(reranked_results)
        trace = legacy_trace(
            intent,
            expanded_queries,
            reranked_results,
            sparse_weight,
            dense_weight,
            candidate_count=len(fused_results),
            provider=generation_provider,
        )
        emit_progress(
            on_event, "generation", "done",
            f"Answer generated by {generation_provider}.",
            provider=generation_provider,
            grounded=False,
            citations=len(citations),
        )

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    logger.log_latency("Total Request", start_time)
    emit_progress(
        on_event, "complete", "done",
        "Retrieval and generation complete.",
        mode=trace.get("mode", "baseline"),
        elapsed_ms=elapsed_ms,
    )
    return {
        "session_id": session_id,
        "response": {
            "answer": answer,
            "citations": citations,
            "context_chunks": context_chunks,
            "abstained": abstained,
            "verification": verification,
        },
        "intent_detected": intent,
        "expanded_queries": expanded_queries,
        "trace": trace,
    }


@app.post("/chat")
def chat(request: QueryRequest):
    """Synchronous chat endpoint retained for clients without SSE support."""
    return _run_chat(request)


@app.post("/chat/stream")
def chat_stream(request: QueryRequest):
    """Stream retrieval and generation progress as Server-Sent Events."""
    events: queue.Queue = queue.Queue()
    sentinel = object()

    def listener(event: dict) -> None:
        events.put({"type": "progress", **event})

    def worker() -> None:
        try:
            result = _run_chat(request, on_event=listener)
            events.put({"type": "final", "data": result})
        except Exception as exc:  # surface a terminal error to the UI
            logger.log_error(f"Streaming chat failed: {exc}")
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(sentinel)

    def event_stream():
        yield "retry: 3000\n\n"
        while True:
            try:
                item = events.get(timeout=15)
            except queue.Empty:
                # Keep proxies and browsers from closing an idle connection while
                # the local generation model is still working.
                yield ": keep-alive\n\n"
                continue
            if item is sentinel:
                break
            payload = json.dumps(item, ensure_ascii=False, default=str)
            yield f"data: {payload}\n\n"

    threading.Thread(target=worker, name="chat-stream", daemon=True).start()
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and index a document (PDF or TXT)."""
    filename = file.filename
    print(f"\n📤 Uploading: {filename}")
    
    try:
        # Validate file
        if not file:
            raise ValueError("No file provided")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save file
        file_path = os.path.join("data", filename)
        content = await file.read()
        
        if not content:
            raise ValueError(f"File is empty: {filename}")
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        print(f"  ✓ File saved to {file_path} ({len(content)} bytes)")
        
        # Index the file
        from starlette.concurrency import run_in_threadpool
        
        def run_indexer():
            try:
                # Get the embedder model from QueryEmbeddingModel
                embedder_instance = embedder.embedder if hasattr(embedder, 'embedder') else embedder
                
                # Create indexer with global storage (already initialized at startup)
                indexer = IndexBuilder(
                    data_dir="data",
                    embedder=embedder_instance,
                    storage=storage  # Use global storage initialized at startup
                )
                
                # Index the specific file
                chunks = indexer.build_index_for_file(file_path)
                
                if chunks is None:
                    raise ValueError(f"Failed to index {filename}")
                
                return chunks
                
            except Exception as e:
                logger.log_error(f"Indexing error for {filename}: {str(e)}")
                raise
        
        # Run indexing in thread pool to avoid blocking
        try:
            chunks = await run_in_threadpool(run_indexer)
        except Exception as e:
            print(f"  ✗ Indexing failed: {str(e)}")
            raise
        
        if not chunks:
            raise ValueError(f"No content extracted from {filename}")
        
        # Update BM25 index
        try:
            texts = [chunk.page_content for chunk in chunks]
            ids = [chunk.metadata.get("chunk_id", "") for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            tfidf.add_documents(texts, ids=ids, metadatas=metadatas)
            print(f"  ✓ BM25 index updated with {len(texts)} texts")
        except Exception as e:
            logger.log_error(f"BM25 update error: {str(e)}")
            # Don't fail completely if BM25 fails, just log it
            print(f"  ⚠ Warning: BM25 update failed: {str(e)}")
        
        # Log success
        logger.log_upload(filename, len(chunks))

        # Build the SAH-RAG section tree without breaking legacy indexing.
        structure_info = None
        if sah_config.structure_enabled:
            try:
                result = ingest_document(file_path, structure_store, existing_chunks=chunks)
                structure_info = {
                    "doc_id": result["doc_id"],
                    "strategy": result["strategy"],
                    "sections": result["sections"],
                    "chunks": result["chunks"],
                }
                print(
                    f"  ✓ Structure: {result['strategy']} | "
                    f"{result['sections']} sections | {result['chunks']} chunks"
                )
            except Exception as e:
                logger.log_error(f"Structure extraction error for {filename}: {str(e)}")
                print(f"  ⚠ Structure extraction failed (legacy index intact): {str(e)}")
        
        print(f"✅ Successfully indexed {filename}: {len(chunks)} chunks\n")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded and indexed {filename}",
            "chunks": len(chunks),
            "filename": filename,
            "structure": structure_info,
        }
        
    except Exception as e:
        error_msg = f"Upload failed for {filename}: {str(e)}"
        print(f"❌ {error_msg}\n")
        logger.log_error(error_msg)
        
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

# ==========================================
# Structure / source explorer endpoints
# ==========================================
@app.get("/documents")
def list_documents():
    """List indexed documents and their section counts."""
    return {"documents": structure_store.list_documents(), "stats": structure_store.stats()}

@app.get("/documents/{doc_id}/structure")
def document_structure(doc_id: str):
    """Return a document's section tree and chunk-to-section mapping."""
    document = structure_store.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    sections = structure_store.get_sections(doc_id)
    chunks = [
        chunk
        for section in sections
        for chunk in structure_store.get_chunks_for_section(section["section_id"])
    ]
    return {"document": document, "sections": sections, "chunks": chunks}

# ==========================================
# Frontend Serving (Monolithic Deployment)
# ==========================================
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# The dist folder is generated by the Docker multi-stage build or `npm run build`
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "ui", "dist")

if os.path.exists(frontend_dist):
    # Mount the /assets directory directly
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
        
    # Catch-all route to serve the SPA index.html for React Router
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Prevent accessing the root index.html recursively by serving the file if it exists
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Fallback to index.html for client-side routing
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    print(f"Warning: Frontend build directory not found at {frontend_dist}. Running in API-only mode.")
