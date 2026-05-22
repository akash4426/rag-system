import os
from typing import List, Dict, Any
from pinecone import Pinecone


class Storage:
    """Component 22: Storage
    Manages vector storage and metadata using Pinecone.
    """
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-system")

        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")

        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Upsert vectors with text stored in metadata."""
        vectors = []
        for i in range(len(ids)):
            meta = {**(metadatas[i] if metadatas[i] else {}), "text": documents[i]}
            # Pinecone metadata values must be strings, numbers, booleans, or lists of strings
            sanitized_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                elif isinstance(v, list):
                    sanitized_meta[k] = [str(item) for item in v]
                else:
                    sanitized_meta[k] = str(v)

            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": sanitized_meta
            })

        # Pinecone upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace="default")

    def get_all_documents(self) -> List[str]:
        """Fetch all document texts from Pinecone for BM25 rebuild.
        Uses list + fetch to retrieve all vectors and their metadata.
        """
        all_texts = []
        try:
            # List all vector IDs in the namespace
            ids_response = self.index.list(namespace="default")
            all_ids = []
            for id_batch in ids_response:
                if isinstance(id_batch, list):
                    all_ids.extend(id_batch)
                else:
                    all_ids.append(id_batch)

            if not all_ids:
                return []

            # Fetch vectors in batches of 100
            for i in range(0, len(all_ids), 100):
                batch_ids = all_ids[i:i + 100]
                fetch_response = self.index.fetch(ids=batch_ids, namespace="default")
                for vec_id, vec_data in fetch_response.vectors.items():
                    text = vec_data.metadata.get("text", "")
                    if text:
                        all_texts.append(text)
        except Exception as e:
            print(f"Failed to fetch documents from Pinecone: {e}")

        return all_texts

    def query(self, query_embedding: List[float], top_k: int = 5) -> list:
        """Query Pinecone for similar vectors."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="default"
            )
            return results.matches
        except Exception as e:
            print(f"Pinecone query failed: {e}")
            return []
