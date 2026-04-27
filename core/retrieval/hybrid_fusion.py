class HybridFusionEngine:
    """Component 9: Hybrid Fusion Engine
    Fuses sparse (BM25) and dense (Vector) retrieval results using Reciprocal Rank Fusion (RRF).
    """
    def fuse(self, sparse_results: list[dict], dense_results: list[dict], sparse_weight: float = 1.0, dense_weight: float = 1.0) -> list[dict]:
        fused_scores = {}
        content_map = {}
        
        # RRF constant
        k = 60
        
        def process_results(results, weight, source_name):
            for rank, result in enumerate(results):
                content = result["content"]
                if content not in fused_scores:
                    fused_scores[content] = 0
                    content_map[content] = {"content": content, "sources": set()}
                
                # RRF Score formula
                score = weight * (1 / (k + rank + 1))
                fused_scores[content] += score
                content_map[content]["sources"].add(source_name)

        process_results(sparse_results, sparse_weight, "bm25")
        process_results(dense_results, dense_weight, "vector_db")
        
        # Sort by fused score
        fused_list = []
        for content, score in fused_scores.items():
            fused_list.append({
                "content": content,
                "score": score,
                "sources": list(content_map[content]["sources"])
            })
            
        fused_list.sort(key=lambda x: x["score"], reverse=True)
        return fused_list
