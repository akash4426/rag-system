from fastapi import FastAPI, HTTPException, UploadFile, File
import os
from dotenv import load_dotenv

# Load environment variables before importing components
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    print("="*60 + "\n")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_db": "pinecone" if storage.use_pinecone else "memory"
    }

@app.post("/chat")
def chat(request: QueryRequest):
    start_time = time.time()
    
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = MemoryModule()
        
    memory = sessions[session_id]
    raw_query = request.query
    logger.log_query(session_id, raw_query)

    # 1. Query Processing
    t0 = time.time()
    clean_query = preprocessor.preprocess(raw_query)
    intent = classifier.classify(clean_query)
    expanded_queries = expander.expand(clean_query)
    logger.log_latency("Query Processing", t0)
    
    # 2. Retrieval
    t0 = time.time()
    sparse_weight, dense_weight = weights.get_weights(intent)
    
    all_sparse_results = []
    all_dense_results = []
    
    for q in expanded_queries:
        # Sparse
        sparse_res = tfidf.retrieve(q)
        all_sparse_results.extend(sparse_res)
        
        # Dense
        q_emb = embedder.embed_query(q)
        dense_res = vector_db.retrieve(q_emb)
        all_dense_results.extend(dense_res)

    # 3. Fusion & Re-ranking
    fused_results = fusion.fuse(all_sparse_results, all_dense_results, sparse_weight, dense_weight)
    reranked_results = reranker.rerank(clean_query, fused_results, top_k=3)
    logger.log_latency("Retrieval & Fusion", t0)

    # 4. Generation
    t0 = time.time()
    context = context_builder.build_context(reranked_results)
    history = memory.get_history_string()
    
    prompt = prompt_builder.build(clean_query, context, history)
    prompt = token_manager.truncate_context(prompt, prompt_template_tokens=500) # Ensure fits
    
    raw_response = llm.generate(prompt)
    
    memory.add_interaction(clean_query, raw_response)
    
    formatted_response = formatter.format(raw_response, reranked_results)
    logger.log_latency("Generation", t0)
    logger.log_generation(session_id, prompt, raw_response)
    logger.log_latency("Total Request", start_time)

    return {
        "session_id": session_id,
        "response": formatted_response,
        "intent_detected": intent,
        "expanded_queries": expanded_queries
    }

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
            tfidf.add_documents(texts)
            print(f"  ✓ BM25 index updated with {len(texts)} texts")
        except Exception as e:
            logger.log_error(f"BM25 update error: {str(e)}")
            # Don't fail completely if BM25 fails, just log it
            print(f"  ⚠ Warning: BM25 update failed: {str(e)}")
        
        # Log success
        logger.log_upload(filename, len(chunks))
        
        print(f"✅ Successfully indexed {filename}: {len(chunks)} chunks\n")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded and indexed {filename}",
            "chunks": len(chunks),
            "filename": filename
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
