from sentence_transformers import CrossEncoder

class Reranker:
    """Component 11: Re-ranker
    Uses a Cross-Encoder to accurately score and re-rank the fused results.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        if not documents:
            return []
            
        # Prepare pairs of (query, document_content)
        pairs = [[query, doc["content"]] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Assign scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
            
        # Sort by rerank score
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return documents[:top_k]
