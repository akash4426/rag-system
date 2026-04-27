from core.ingestion.storage import Storage

class VectorDBRetriever:
    """Component 8: Vector DB Retriever
    Performs dense retrieval using the query embedding.
    """
    def __init__(self):
        self.storage = Storage()

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        try:
            results = self.storage.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            retrieved_docs = []
            if results and results['documents'] and len(results['documents']) > 0:
                docs = results['documents'][0]
                distances = results['distances'][0]
                
                for i, doc in enumerate(docs):
                    retrieved_docs.append({
                        "content": doc,
                        "score": float(distances[i]), # L2 distance typically
                        "source": "vector_db"
                    })
                    
            return retrieved_docs
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []
