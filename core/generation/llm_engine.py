import os
from langchain_openai import ChatOpenAI

class LLMEngine:
    """Component 15: LLM Engine
    Executes the final generation using the LLM (OpenRouter).
    """
    def __init__(self):
        api_key = os.getenv("OPEN_ROUTER_API")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        if not api_key:
            print("WARNING: OPEN_ROUTER_API not set. Generation will fail.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                model_name=model,
                temperature=0.3
            )

    def generate(self, prompt: str) -> str:
        if not self.llm:
            return "Error: LLM Engine not configured."
            
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error during generation: {e}"
