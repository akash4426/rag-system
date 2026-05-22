import uuid
from core.ingestion.loader import DocumentLoader
from core.ingestion.splitter import TextSplitter
from core.ingestion.embedder import EmbeddingGenerator
from core.ingestion.storage import Storage


class IndexBuilder:
    """Component 21: Index Builder
    Orchestrates the ingestion pipeline. Stores vectors in Pinecone.
    BM25 corpus is rebuilt in-memory from Pinecone on startup.
    """
    def __init__(self, data_dir: str = "data"):
        self.loader = DocumentLoader(directory=data_dir)
        self.splitter = TextSplitter()
        self.embedder = EmbeddingGenerator()
        self.storage = Storage()

    def build_index(self):
        print("Starting Indexing Process...")
        # 1. Load documents
        docs = self.loader.load_documents()
        if not docs:
            print("No documents found in data directory.")
            return

        # 2. Split documents
        chunks = self.splitter.split(docs)

        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)

        # 3. Embed documents
        embeddings = self.embedder.generate_embeddings_batch(texts)

        # 4. Store in Pinecone
        self.storage.add_documents(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        print(f"Successfully indexed {len(chunks)} chunks into Pinecone.")

    def build_index_for_file(self, filepath: str):
        print(f"Starting Incremental Indexing for {filepath}...")
        docs = self.loader.load_single_document(filepath)
        if not docs:
            print("Failed to load document.")
            return

        chunks = self.splitter.split(docs)
        if not chunks:
            return

        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)

        embeddings = self.embedder.generate_embeddings_batch(texts)
        self.storage.add_documents(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        print(f"Successfully incrementally indexed {len(chunks)} chunks from {filepath}.")

if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build_index()
