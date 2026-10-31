"""Retrieval packages for SAH-RAG.

The legacy retrievers live in :mod:`core.retrieval`; everything here wraps or
extends them without replacing FAISS/BM25/baseline behaviour.
"""

from retrieval.types import Chunk

__all__ = ["Chunk"]
