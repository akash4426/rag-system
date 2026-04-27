import tiktoken

class TokenManager:
    """Component 13: Token Manager
    Ensures that the combined prompt + context does not exceed LLM context window limits.
    """
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # We use tiktoken for general estimation
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.max_tokens = 8000 # Configurable limit

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_context(self, context: str, prompt_template_tokens: int) -> str:
        # If context is too large, truncate it
        tokens = self.encoding.encode(context)
        allowed_tokens = self.max_tokens - prompt_template_tokens
        
        if len(tokens) <= allowed_tokens:
            return context
            
        truncated_tokens = tokens[:allowed_tokens]
        return self.encoding.decode(truncated_tokens)
