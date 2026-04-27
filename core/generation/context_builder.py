class ContextBuilder:
    """Component 12: Context Builder
    Assembles the final context string from the re-ranked documents.
    """
    def build_context(self, documents: list[dict]) -> str:
        if not documents:
            return "No relevant context found."
            
        context_parts = []
        for i, doc in enumerate(documents):
            # Include a source tag or just the content
            context_parts.append(f"[Document {i+1}]:\n{doc['content']}")
            
        return "\n\n".join(context_parts)
