from core.ingestion.embedder import EmbeddingGenerator

class QueryEmbeddingModel:
    """Component 7: Embedding Model for Query
    Wrapper to generate embeddings for the incoming query at runtime.
    """
    def __init__(self, embedder=None):
        # Reuse the ingestion embedder when one is supplied.
        self.embedder = embedder if embedder is not None else EmbeddingGenerator()

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.generate_embedding(query)
