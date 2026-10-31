import os
import uuid

from core.ingestion.loader import DocumentLoader
from core.ingestion.splitter import TextSplitter
from core.ingestion.embedder import EmbeddingGenerator
from core.ingestion.storage import Storage


class IndexBuilder:
    """Component 21: Index Builder
    Orchestrates the ingestion pipeline. Stores vectors in Pinecone.
    BM25 corpus is rebuilt in-memory from Pinecone on startup.

    SAH-RAG adds stable chunk IDs and document/page provenance to metadata;
    the original split/embed/store flow is unchanged.
    """
    def __init__(self, data_dir: str = "data", embedder=None, storage=None):
        self.loader = DocumentLoader(directory=data_dir)
        self.splitter = TextSplitter()
        self.embedder = embedder if embedder else EmbeddingGenerator()
        self.storage = storage if storage else Storage()

    def _prepare_chunks(self, chunks, filepath: str):
        """Attach chunk_id/doc/page metadata and return serializable arrays."""
        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            metadata = dict(chunk.metadata or {})
            metadata["chunk_id"] = chunk_id
            metadata["doc"] = os.path.basename(filepath)
            if "page" in metadata and metadata["page"] is not None:
                try:
                    metadata["page_number"] = int(metadata["page"]) + 1
                except (TypeError, ValueError):
                    metadata["page_number"] = None
            metadata["source"] = metadata.get("source", filepath)

            chunk.metadata = metadata
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            metadatas.append(metadata)

        return ids, texts, metadatas

    def build_index(self):
        print("Starting Indexing Process...")
        # 1. Load documents
        docs = self.loader.load_documents()
        if not docs:
            print("No documents found in data directory.")
            return

        # 2. Split documents
        chunks = self.splitter.split(docs)

        # 3. Attach provenance
        ids, texts, metadatas = self._prepare_chunks(chunks, "data")

        # 4. Embed documents
        embeddings = self.embedder.generate_embeddings_batch(texts)

        # 5. Store in Pinecone
        self.storage.add_documents(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        print(f"Successfully indexed {len(chunks)} chunks into Pinecone.")
        return chunks

    def build_index_for_file(self, filepath: str):
        print(f"Starting Incremental Indexing for {filepath}...")
        docs = self.loader.load_single_document(filepath)
        if not docs:
            print("Failed to load document.")
            return

        chunks = self.splitter.split(docs)
        if not chunks:
            return

        ids, texts, metadatas = self._prepare_chunks(chunks, filepath)
        embeddings = self.embedder.generate_embeddings_batch(texts)
        self.storage.add_documents(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        print(f"Successfully incrementally indexed {len(chunks)} chunks from {filepath}.")
        return chunks


if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build_index()
