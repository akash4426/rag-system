from core.ingestion.embedder import EmbeddingGenerator

class QueryEmbeddingModel:
    """Component 7: Embedding Model for Query
    Wrapper to generate embeddings for the incoming query at runtime.
    """
    def __init__(self):
        # Reuse the ingestion embedder
        self.embedder = EmbeddingGenerator()

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.generate_embedding(query)
