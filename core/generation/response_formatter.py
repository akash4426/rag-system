class ResponseFormatter:
    """Component 16: Response Formatter
    Formats the raw LLM output into a clean, presentation-ready format (e.g., adding citations).
    """
    def format(self, raw_response: str, sources: list[dict] = None) -> dict:
        formatted_response = {
            "answer": raw_response,
            "citations": [],
            "context_chunks": []
        }
        
        if sources:
            # We map back the source metadata and the actual text
            formatted_response["citations"] = [", ".join(s.get("sources", ["Unknown"])) for s in sources]
            formatted_response["context_chunks"] = [
                {
                    "source": ", ".join(s.get("sources", ["Unknown"])),
                    "text": s.get("content", "")
                } 
                for s in sources
            ]
            
        return formatted_response
