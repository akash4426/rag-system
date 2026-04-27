import re

class IntentClassifier:
    """Component 4: Intent Classifier
    Classifies the user query intent to guide the retrieval strategy.
    """
    def __init__(self):
        # Basic heuristic-based classification for speed
        self.keywords = {
            "summarization": ["summarize", "summary", "tl;dr", "tldr", "briefly"],
            "factual": ["who", "what", "where", "when", "how many", "which"],
            "analytical": ["why", "how does", "explain", "compare", "difference"]
        }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        
        for intent, words in self.keywords.items():
            for word in words:
                if re.search(r'\b' + word + r'\b', query_lower):
                    return intent
        
        return "general"
