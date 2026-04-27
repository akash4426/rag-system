import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

class Storage:
    """Component 22: Storage
    Manages vector storage and metadata using ChromaDB.
    """
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def get_all_documents(self):
        return self.collection.get()
