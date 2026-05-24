import os
from typing import List, Dict, Any

class Storage:
    """Component 22: Storage
    Manages vector storage and metadata using Pinecone as the primary backend.
    Provides a unified interface for document ingestion and vector retrieval.
    
    Configuration:
    - Uses Pinecone by default (requires PINECONE_API_KEY and PINECONE_ENVIRONMENT)
    - Falls back to in-memory storage if Pinecone is not configured
    """
    
    def __init__(self, use_pinecone: bool = True):
        """Initialize storage backend.
        
        Args:
            use_pinecone: Whether to use Pinecone (default: True)
        """
        self.use_pinecone = use_pinecone and os.getenv("PINECONE_API_KEY") is not None
        self.backend = None
        
        if self.use_pinecone:
            try:
                from core.ingestion.pinecone_storage import PineconeStorage
                self.backend = PineconeStorage()
                print("✓ Using Pinecone as vector storage backend")
            except Exception as e:
                print(f"⚠ Pinecone initialization failed: {e}")
                print("⚠ Falling back to in-memory storage")
                self.use_pinecone = False
                self._init_memory_storage()
        else:
            self._init_memory_storage()
    
    def _init_memory_storage(self):
        """Initialize in-memory storage as fallback."""
        self.memory_vectors = {}  # id -> {embedding, metadata, text}
        self.memory_texts = []
        print("✓ Using in-memory storage backend")

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Upsert vectors with metadata."""
        if self.use_pinecone:
            self.backend.add_documents(ids, embeddings, documents, metadatas)
        else:
            # In-memory fallback
            for doc_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
                safe_metadata = metadata if metadata else {}
                safe_metadata["text"] = document
                self.memory_vectors[doc_id] = {
                    "embedding": embedding,
                    "metadata": safe_metadata,
                    "text": document
                }
                self.memory_texts.append(document)
            print(f"Added {len(ids)} documents to in-memory storage")

    def get_all_documents(self) -> List[str]:
        """Fetch all document texts for BM25 rebuild."""
        if self.use_pinecone:
            return self.backend.get_all_documents()
        else:
            return self.memory_texts.copy()

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        """Query for similar vectors."""
        if self.use_pinecone:
            return self.backend.query(query_embedding, top_k)
        else:
            # Simple in-memory cosine similarity search
            import numpy as np
            
            if not self.memory_vectors:
                return []
            
            query_vec = np.array(query_embedding)
            scores = []
            
            for doc_id, data in self.memory_vectors.items():
                embedding_vec = np.array(data["embedding"])
                # Cosine similarity
                similarity = np.dot(query_vec, embedding_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(embedding_vec) + 1e-8
                )
                scores.append({
                    "id": doc_id,
                    "score": float(similarity),
                    "metadata": data["metadata"]
                })
            
            # Sort by score and return top_k
            scores.sort(key=lambda x: x["score"], reverse=True)
            return scores[:top_k]

    def delete_documents(self, ids: List[str]):
        """Delete documents by ID."""
        if self.use_pinecone:
            self.backend.delete_documents(ids)
        else:
            for doc_id in ids:
                if doc_id in self.memory_vectors:
                    text = self.memory_vectors[doc_id].get("text", "")
                    del self.memory_vectors[doc_id]
                    if text in self.memory_texts:
                        self.memory_texts.remove(text)
            print(f"Deleted {len(ids)} documents from in-memory storage")

    def clear_index(self):
        """Clear all vectors."""
        if self.use_pinecone:
            self.backend.clear_index()
        else:
            self.memory_vectors.clear()
            self.memory_texts.clear()
            print("Cleared in-memory storage")
