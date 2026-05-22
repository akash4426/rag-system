from rank_bm25 import BM25Okapi
from core.ingestion.storage import Storage


class TFIDFRetriever:
    """Component 6: TF-IDF / BM25 Retriever
    Performs sparse retrieval using BM25 on the document corpus.
    Rebuilds the BM25 index in-memory from Pinecone on startup.
    """
    def __init__(self):
        self.corpus = []
        self.bm25 = None
        self._storage = None
        self._build_from_pinecone()

    def _get_storage(self):
        """Lazy init storage to avoid circular imports."""
        if self._storage is None:
            self._storage = Storage()
        return self._storage

    def _build_from_pinecone(self):
        """Fetch all document texts from Pinecone and build BM25 index."""
        try:
            storage = self._get_storage()
            self.corpus = storage.get_all_documents()

            if self.corpus:
                tokenized_corpus = [doc.lower().split() for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"BM25 index built with {len(self.corpus)} documents from Pinecone.")
            else:
                print("No documents found in Pinecone for BM25 index.")
        except Exception as e:
            print(f"Failed to build BM25 from Pinecone: {e}")
            self.corpus = []
            self.bm25 = None

    def reload_corpus(self):
        """Rebuild BM25 index from Pinecone (called after new document upload)."""
        self._build_from_pinecone()

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
                    "source": "bm25"
                })

        return results
