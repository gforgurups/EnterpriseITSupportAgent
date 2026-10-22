"""
Process Documents module
1. Text Extraction
2. Text Cleaning
3. Text Chunking
"""

from typing import List, Union
from langchain_community.document_loaders import (
            WebBaseLoader,
            PyPDFLoader,
            TextLoader,
            PyPDFDirectoryLoader
            )
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from docx import Document as DocxDocument

class DocumentProcessor:
    """ Main class to load and process documents"""

    def __init__(self, chunk_size=900, chunk_overlap=120):
        """ Initialize DocumentProcessor with chunk size and overlap"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )   

    def load_web_documents(self, urls: List[str]) -> List[Document]:
        """ Load web documents from URLs and return raw documents"""
        docs = []
        for url in urls:
            docs.extend(WebBaseLoader(url).load())
        return docs

    def load_docx_document(self, filepath: Union[str, Path]) -> List[Document]:
        loader = DocxDocument(filepath)
        text = "\n".join([p.text for p in loader.paragraphs if p.text.strip()])
        return [Document(page_content=text, metadata={"source":filepath})]

    #filepath: Union[str, Path] meansvalue can be one of multiple allowed types.
    def load_pdf_document(self, filepath: Union[str, Path]) -> List[Document]:
        """Load PDF documents from filepaths and return raw documents"""
        loader = PyPDFLoader(filepath)
        return loader.load()

    def load_pdf_directory_documents(self, directory: Union[str, Path], glob: str = "*.pdf") -> List[Document]:
        """Load documents from a directory and return raw documents"""
        loader = PyPDFDirectoryLoader(directory, glob=glob)
        return loader.load()

    def load_text_document(self, filepath: Union[str,Path]) -> List[Document]:
        """Load text documents from filepaths and return raw documents"""
        loader = TextLoader(filepath,encoding="utf-8")
        return loader.load()

    def load_documents(self, sources:List[str]) -> List[Document]:
        """Load documents from multiple sources and return raw documents"""
        all_docs = []
        for source in sources:
            if source.startswith("http"):
                all_docs.extend(self.load_web_documents([source]))
            elif source.endswith(".pdf"):
                all_docs.extend(self.load_pdf_document(source))
            elif source.endswith((".txt",".md")):
                all_docs.extend(self.load_text_document(source))
            elif source.endswith(".docx"):
                all_docs.extend(self.load_docx_document(source))
            else:
                raise ValueError(f"Unsupported file type")
        return all_docs
        
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks and return chunks"""
        return self.splitter.split_documents(docs)

    def ingest_documents(self, sources: List[str]) -> List[Document]:
        """Ingest documents from multiple sources and return chunks"""
        docs = self.load_documents(sources)     #Returns raw documents 
        chunks = self.split_documents(docs)     #Splits documents into chunks and returns chunks
        return chunks
    
        

    