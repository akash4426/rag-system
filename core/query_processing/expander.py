import os
from langchain_openai import ChatOpenAI

class QueryExpander:
    """Component 5: Query Expander
    Expands the original query to include synonyms or sub-queries to improve recall.
    """
    def __init__(self):
        api_key = os.getenv("OPEN_ROUTER_API")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        if api_key:
            self.llm = ChatOpenAI(
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                model_name=model,
                temperature=0.2
            )
        else:
            self.llm = None

    def expand(self, query: str) -> list[str]:
        if not self.llm:
            return [query] # Fallback if no LLM
            
        prompt = f"Given the user query: '{query}', generate 2 alternative ways to ask this question or relevant sub-queries that would help a search engine find relevant documents. Return ONLY the queries separated by a newline."
        
        try:
            response = self.llm.invoke(prompt)
            expanded = response.content.split('\n')
            expanded = [q.strip() for q in expanded if q.strip()]
            return [query] + expanded # Include original query
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]
