class MemoryModule:
    """Component 17: Memory Module
    Maintains conversation history for multi-turn RAG interactions.
    """
    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history

    def add_interaction(self, query: str, response: str):
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_history_string(self) -> str:
        if not self.history:
            return "No previous history."
            
        history_strs = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_strs.append(f"{role}: {msg['content']}")
            
        return "\n".join(history_strs)
    
    def clear(self):
        self.history = []
