class DynamicWeightController:
    """Component 10: Dynamic Weight Controller
    Adjusts the weights of sparse vs dense retrieval based on query intent.
    """
    def get_weights(self, intent: str) -> tuple[float, float]:
        # Returns (sparse_weight, dense_weight)
        if intent == "factual":
            # Factual queries benefit from exact keyword matches
            return (1.5, 1.0)
        elif intent == "analytical" or intent == "summarization":
            # Semantic search is better for conceptual queries
            return (0.8, 1.5)
        else:
            # Default balanced
            return (1.0, 1.0)
