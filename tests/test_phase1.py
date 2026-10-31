"""Phase 1 tests: metrics and typed baseline retrieval."""

from eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, token_f1
from retrieval.baseline import BaselineRetriever
from retrieval.types import Chunk


class FakeBM25:
    def retrieve(self, query, top_k=5):
        return [
            {"content": "alpha beta", "score": 2.0, "source": "bm25", "chunk_id": "c1",
             "metadata": {"doc": "a.pdf", "page_number": 3}},
            {"content": "gamma", "score": 1.0, "source": "bm25", "chunk_id": "c2",
             "metadata": {"doc": "a.pdf", "page_number": 4}},
        ][:top_k]


class FakeDense:
    def retrieve(self, embedding, top_k=5):
        return [
            {"content": "delta", "score": 0.9, "source": "vector_db", "chunk_id": "c3",
             "metadata": {"doc": "a.pdf", "page_number": 5}},
            {"content": "alpha beta", "score": 0.8, "source": "vector_db", "chunk_id": "c1",
             "metadata": {"doc": "a.pdf", "page_number": 3}},
        ][:top_k]


class FakeEmbedder:
    def embed_query(self, query):
        return [0.1, 0.2]


def test_metrics_binary_relevance():
    flags = [False, True, False, True]
    assert recall_at_k(flags, 2) == 0.5
    assert precision_at_k(flags, 4) == 0.5
    assert reciprocal_rank(flags) == 0.5
    assert 0 < ndcg_at_k(flags, 4) <= 1.0


def test_metrics_empty_and_token_f1():
    assert recall_at_k([], 5) == 0.0
    assert precision_at_k([], 5) == 0.0
    assert reciprocal_rank([False, False]) == 0.0
    assert token_f1("the quick fox", "the quick fox") == 1.0
    assert token_f1("", "x") == 0.0


def test_baseline_fuses_without_duplicates():
    retriever = BaselineRetriever(
        tfidf=FakeBM25(), vector_db=FakeDense(), embedder=FakeEmbedder()
    )
    chunks = retriever.retrieve_baseline("alpha", k=3)
    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c3", "c2"]
    assert chunks[0].sources == ["bm25", "vector_db"]
    assert chunks[0].page == 3
    assert chunks[0].doc == "a.pdf"
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].metadata["fused_score"] > chunks[1].metadata["fused_score"]
