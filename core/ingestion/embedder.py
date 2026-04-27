from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Component 20: Embedding Generator
    Generates dense vector embeddings for text chunks.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # We use a lightweight local model for fast dense embeddings
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
