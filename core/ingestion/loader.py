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
        
        if not os.path.exists(self.directory):
            print(f"Warning: Directory not found: {self.directory}")
            return documents
        
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            
            # Skip directories
            if os.path.isdir(filepath):
                continue
            
            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"  ✓ Loaded {filename}: {len(docs)} page(s)")
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"  ✓ Loaded {filename}: {len(docs)} page(s)")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to load {filename}: {str(e)}")
                # Continue with other files
                continue
        
        return documents

    def load_single_document(self, filepath: str) -> List[Document]:
        """Load a single document from a file path."""
        documents = []
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not os.path.isfile(filepath):
            raise ValueError(f"Path is not a file: {filepath}")
        
        try:
            if filepath.endswith(".txt"):
                loader = TextLoader(filepath, encoding='utf-8')
                documents = loader.load()
            elif filepath.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            if not documents:
                raise ValueError(f"No content extracted from {filepath}")
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to load document {filepath}: {str(e)}")
