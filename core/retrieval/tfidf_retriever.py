import os
import pickle
from rank_bm25 import BM25Okapi

class TFIDFRetriever:
    """Component 6: TF-IDF / BM25 Retriever
    Performs sparse retrieval using BM25 on the document corpus.
    """
    def __init__(self, corpus_path: str = "bm25_corpus.pkl"):
        self.corpus_path = corpus_path
        self.corpus = []
        self.bm25 = None
        
        if os.path.exists(corpus_path):
            with open(corpus_path, 'rb') as f:
                self.corpus = pickle.load(f)
            
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)

    def reload_corpus(self):
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, 'rb') as f:
                self.corpus = pickle.load(f)
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)

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
