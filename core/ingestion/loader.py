import os
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

class DocumentLoader:
    """Component 18: Document Loader
    Responsible for ingesting raw documents from various formats.
    """
    def __init__(self, directory: str = "data"):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def load_documents(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding='utf-8')
                documents.extend(loader.load())
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
            # Can be extended for other formats (CSV, DOCX, etc.)
        return documents

    def load_single_document(self, filepath: str) -> List[Document]:
        documents = []
        if filepath.endswith(".txt"):
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
        elif filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        return documents
