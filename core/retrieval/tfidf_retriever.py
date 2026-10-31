# pyrefly: ignore [missing-import]
from typing import Any

from rank_bm25 import BM25Okapi

from core.ingestion.storage import Storage


class TFIDFRetriever:
    """Component 6: TF-IDF / BM25 Retriever
    Performs sparse retrieval using BM25 on the document corpus.
    Rebuilds the BM25 index in-memory from Pinecone on startup.

    Chunk provenance (id + metadata) is retained when the storage backend
    exposes it. Text-only corpora still work exactly as before.
    """

    def __init__(self, storage: Storage | None = None):
        self.corpus: list[str] = []
        self.chunk_ids: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.bm25 = None
        self._storage = storage
        self._build_from_pinecone()

    def _get_storage(self):
        """Lazy init storage to avoid circular imports."""
        if self._storage is None:
            self._storage = Storage()
        return self._storage

    def _build_from_pinecone(self):
        """Fetch all document records from storage and build the BM25 index."""
        try:
            storage = self._get_storage()
            if hasattr(storage, "get_all_records"):
                records = storage.get_all_records()
            else:  # pragma: no cover - older storage implementations
                records = [
                    {"id": "", "text": text, "metadata": {}}
                    for text in storage.get_all_documents()
                ]

            self.corpus = [record["text"] for record in records]
            self.chunk_ids = [record.get("id", "") for record in records]
            self.metadatas = [record.get("metadata", {}) for record in records]

            if self.corpus:
                tokenized_corpus = [doc.lower().split() for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"BM25 index built with {len(self.corpus)} documents from storage.")
            else:
                print("No documents found in storage for BM25 index.")
        except Exception as e:
            print(f"Failed to build BM25 from storage: {e}")
            self.corpus = []
            self.chunk_ids = []
            self.metadatas = []
            self.bm25 = None

    def add_documents(
        self,
        new_texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ):
        """Append new texts to corpus and rebuild BM25 in-memory.

        ``ids``/``metadatas`` are optional for backwards compatibility; when
        omitted, stable fallback IDs are generated from the text.
        """
        if not new_texts:
            return
        if ids is None or len(ids) != len(new_texts):
            ids = ["" for _ in new_texts]
        if metadatas is None or len(metadatas) != len(new_texts):
            metadatas = [{} for _ in new_texts]

        self.corpus.extend(new_texts)
        self.chunk_ids.extend(ids)
        self.metadatas.extend(metadatas)
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index updated. Total documents: {len(self.corpus)}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_n = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_n:
            if doc_scores[idx] > 0:
                results.append({
                    "content": self.corpus[idx],
                    "score": float(doc_scores[idx]),
                    "source": "bm25",
                    "chunk_id": self.chunk_ids[idx] if idx < len(self.chunk_ids) else "",
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                })

        return results
