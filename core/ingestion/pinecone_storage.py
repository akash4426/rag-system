import os
from typing import List, Dict, Any
from pinecone import Pinecone
from pinecone import ServerlessSpec


class PineconeStorage:
    """Component 22: Pinecone Storage
    Manages vector storage and metadata using Pinecone as the vector database.
    Replaces ChromaDB with Pinecone for production-grade vector search.
    """
    
    def __init__(self, index_name: str = "rag-index"):
        """Initialize Pinecone client and index.
        
        Environment variables required:
        - PINECONE_API_KEY: Your Pinecone API key
        - PINECONE_ENVIRONMENT: Cloud region (e.g., 'us-east-1')
        - PINECONE_INDEX_NAME: Name of the index (default: 'rag-index')
        """
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", index_name)
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Get or create index
        self._ensure_index_exists()
        
        # Get index reference
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            # List existing indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                print(f"Creating Pinecone index: {self.index_name}")
                # Create index with serverless configuration
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Matches sentence-transformers default embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                print(f"Index {self.index_name} created successfully")
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Error ensuring index exists: {e}")
            raise

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Upsert vectors with metadata to Pinecone."""
        try:
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            
            for i, (doc_id, embedding, document, metadata) in enumerate(zip(ids, embeddings, documents, metadatas)):
                # Ensure metadata is clean
                safe_metadata = metadata if metadata else {}
                safe_metadata = {k: str(v) for k, v in safe_metadata.items() if v is not None}
                
                # Add text to metadata
                safe_metadata["text"] = document
                safe_metadata["document"] = document
                
                vectors_to_upsert.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": safe_metadata
                })
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace="default")
                print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
            
            print(f"Successfully upserted {len(vectors_to_upsert)} documents to Pinecone")
        except Exception as e:
            print(f"Pinecone upsert failed: {e}")
            raise

    def get_all_documents(self) -> List[str]:
        """Fetch all document texts from Pinecone for BM25 rebuild."""
        all_texts = []
        try:
            # Query stats to understand index size
            stats = self.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            print(f"Fetching {total_vectors} vectors from Pinecone for BM25 indexing")
            
            # For large indexes, we'd need pagination. For now, list vectors.
            # Note: Pinecone's list API has limitations; use query as workaround
            if total_vectors == 0:
                return []
            
            # Fetch vectors using a dummy query to get metadata
            # A better approach would be to maintain a separate BM25 corpus
            # or use Pinecone's metadata filtering
            dummy_embedding = [0.0] * 384  # Dummy embedding matching dimension
            
            results = self.index.query(
                vector=dummy_embedding,
                top_k=min(10000, total_vectors),  # Pinecone has query limits
                include_metadata=True,
                namespace="default"
            )
            
            for match in results.matches:
                if "text" in match.metadata:
                    all_texts.append(match.metadata["text"])
                elif "document" in match.metadata:
                    all_texts.append(match.metadata["document"])
            
            print(f"Retrieved {len(all_texts)} texts from Pinecone")
            
        except Exception as e:
            print(f"Failed to fetch documents from Pinecone: {e}")
        
        return all_texts

    def query(self, query_embedding: List[float], top_k: int = 5, namespace: str = "default") -> list:
        """Query Pinecone for similar vectors."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            # Convert Pinecone format to dictionary format
            formatted_matches = []
            for match in results.matches:
                formatted_match = {
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": dict(match.metadata) if match.metadata else {}
                }
                formatted_matches.append(formatted_match)
            
            return formatted_matches
            
        except Exception as e:
            print(f"Pinecone query failed: {e}")
            return []

    def delete_documents(self, ids: List[str], namespace: str = "default"):
        """Delete documents from Pinecone by ID."""
        try:
            self.index.delete(ids=ids, namespace=namespace)
            print(f"Deleted {len(ids)} documents from Pinecone")
        except Exception as e:
            print(f"Pinecone delete failed: {e}")

    def clear_index(self, namespace: str = "default"):
        """Clear all vectors from the index."""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            print(f"Cleared Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Pinecone clear failed: {e}")
