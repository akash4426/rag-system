from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

app = FastAPI(title="23-Component RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for local dev
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

# Simple in-memory session store
sessions = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

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
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Incrementally build the index for the new file
        indexer = IndexBuilder()
        indexer.build_index_for_file(file_path)
        
        # Reload the tfidf retriever corpus so it has the newest BM25 data
        tfidf.reload_corpus()
        
        return {"status": "success", "message": f"Successfully uploaded and indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Serve React Frontend (Unified Deployment) ──
ui_dist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "dist")

if os.path.exists(ui_dist_path):
    # Mount the assets directory explicitly
    assets_path = os.path.join(ui_dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
        
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Serve specific requested files if they exist (favicon, manifest, etc.)
        file_path = os.path.join(ui_dist_path, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fallback to index.html for React SPA routing
        index_path = os.path.join(ui_dist_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend build not found."}
