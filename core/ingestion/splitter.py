from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    """Component 19: Text Splitter
    Splits large documents into smaller chunks for embedding and retrieval.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
