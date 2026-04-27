class PromptBuilder:
    """Component 14: Prompt Builder
    Constructs the final prompt combining instructions, memory, context, and query.
    """
    def build(self, query: str, context: str, history: str = "") -> str:
        prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Conversation History:
{history}

User Question: {query}
Answer:"""
        return prompt
