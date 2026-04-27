import uuid
import pickle
import os
from core.ingestion.loader import DocumentLoader
from core.ingestion.splitter import TextSplitter
from core.ingestion.embedder import EmbeddingGenerator
from core.ingestion.storage import Storage

class IndexBuilder:
    """Component 21: Index Builder
    Orchestrates the ingestion pipeline and builds the sparse (BM25) index.
    """
    def __init__(self, data_dir: str = "data"):
        self.loader = DocumentLoader(directory=data_dir)
        self.splitter = TextSplitter()
        self.embedder = EmbeddingGenerator()
        self.storage = Storage()
        self.bm25_path = "bm25_corpus.pkl"

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

        # 4. Store in Vector DB
        self.storage.add_documents(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        
        # 5. Save Corpus for BM25 (Sparse Index)
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, 'rb') as f:
                existing_texts = pickle.load(f)
            texts = existing_texts + texts

        with open(self.bm25_path, 'wb') as f:
            pickle.dump(texts, f)
            
        print(f"Successfully indexed {len(chunks)} chunks into VectorDB and BM25 Corpus.")

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

        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, 'rb') as f:
                existing_texts = pickle.load(f)
            texts = existing_texts + texts

        with open(self.bm25_path, 'wb') as f:
            pickle.dump(texts, f)

        print(f"Successfully incrementally indexed {len(chunks)} chunks from {filepath}.")
        
if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build_index()
