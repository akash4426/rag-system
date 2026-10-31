from core.ingestion.storage import Storage


class VectorDBRetriever:
    """Component 8: Vector DB Retriever
    Performs dense retrieval using the query embedding against Pinecone.

    Kept as the baseline dense retriever; SAH-RAG never replaces it.
    """

    def __init__(self, storage: Storage | None = None):
        self.storage = storage if storage else Storage()

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        try:
            matches = self.storage.query(query_embedding, top_k=top_k)

            retrieved_docs = []
            for match in matches:
                if isinstance(match, dict):
                    metadata = match.get("metadata", {})
                    score = float(match.get("score", 0.0))
                    match_id = match.get("id", "")
                else:
                    metadata = match.metadata if hasattr(match, 'metadata') else {}
                    score = float(match.score) if hasattr(match, 'score') else 0.0
                    match_id = getattr(match, "id", "")

                text = metadata.get("text", "") if metadata else ""
                retrieved_docs.append({
                    "content": text,
                    "score": score,
                    "source": "vector_db",
                    "chunk_id": metadata.get("chunk_id", match_id) if metadata else match_id,
                    "metadata": dict(metadata) if metadata else {},
                })

            return retrieved_docs
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []
