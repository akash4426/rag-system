import logging
import time

class SystemLogger:
    """Component 23: Logging + Evaluation
    Handles observability, tracking latency, and logging inputs/outputs for offline evaluation.
    """
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("rag_system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RAGSystem")

    def log_query(self, session_id: str, query: str):
        self.logger.info(f"[Session: {session_id}] New Query: {query}")

    def log_latency(self, component: str, start_time: float):
        elapsed = time.time() - start_time
        self.logger.info(f"[{component}] Latency: {elapsed:.4f}s")

    def log_generation(self, session_id: str, prompt: str, response: str):
        self.logger.info(f"[Session: {session_id}] Generated Response Length: {len(response)} chars")

    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
    
    def log_upload(self, filename: str, chunks: int):
        """Log successful document upload."""
        self.logger.info(f"[Upload] Successfully indexed {filename}: {chunks} chunks")
