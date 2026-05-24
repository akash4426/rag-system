from core.ingestion.storage import Storage


class VectorDBRetriever:
    """Component 8: Vector DB Retriever
    Performs dense retrieval using the query embedding against Pinecone.
    """
    def __init__(self):
        self.storage = Storage()

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        try:
            matches = self.storage.query(query_embedding, top_k=top_k)

            retrieved_docs = []
            for match in matches:
                if isinstance(match, dict):
                    metadata = match.get("metadata", {})
                    score = float(match.get("score", 0.0))
                else:
                    metadata = match.metadata if hasattr(match, 'metadata') else {}
                    score = float(match.score) if hasattr(match, 'score') else 0.0
                    
                text = metadata.get("text", "") if metadata else ""
                retrieved_docs.append({
                    "content": text,
                    "score": score,
                    "source": "vector_db"
                })

            return retrieved_docs
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []
