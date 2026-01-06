"""
Document Loader for RAG System
Handles PDF, TXT, and DOCX files with chunking and ChromaDB indexing.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# PDF Processing
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """Represents a document chunk."""
    content: str
    metadata: Dict[str, Any]
    source: str
    page: int = 0
    chunk_id: int = 0


class DocumentLoader:
    """
    Loads and processes documents for RAG.
    
    Supports:
    - PDF files
    - TXT files
    - Chunking with overlap
    - ChromaDB indexing
    """
    
    def __init__(
        self,
        chromadb_path: str = "Datasets/chromadb_umls",
        collection_name: str = "clinical_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize DocumentLoader.
        
        Args:
            chromadb_path: Path to ChromaDB database
            collection_name: Name of collection for documents
            embedding_model: Model for generating embeddings
        """
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        print(f"⏳ Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        print(f"✅ DocumentLoader initialized")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and extract text from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        if PdfReader is None:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        
        documents = []
        reader = PdfReader(file_path)
        filename = Path(file_path).name
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    content=text.strip(),
                    metadata={"source": filename, "page": i + 1, "type": "pdf"},
                    source=filename,
                    page=i + 1
                )
                documents.append(doc)
        
        print(f"📄 Loaded {len(documents)} pages from {filename}")
        return documents
    
    def load_txt(self, file_path: str) -> List[Document]:
        """
        Load text file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            List containing single Document
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = Path(file_path).name
        doc = Document(
            content=content.strip(),
            metadata={"source": filename, "type": "txt"},
            source=filename
        )
        
        print(f"📄 Loaded {filename}")
        return [doc]
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load file based on extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document objects
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        chunks = []
        chunk_id = 0
        
        for doc in documents:
            text = doc.content
            start = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()
                
                if len(chunk_text) > 100:  # Skip very small chunks
                    chunk_doc = Document(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": chunk_id,
                            "start_char": start,
                            "end_char": end
                        },
                        source=doc.source,
                        page=doc.page,
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk_doc)
                    chunk_id += 1
                
                if end >= len(text):
                    break
                start = end - overlap
        
        print(f"📦 Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def index_documents(self, documents: List[Document]) -> int:
        """
        Index documents into ChromaDB.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0
        
        # Prepare data for ChromaDB
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{doc.source}_{doc.chunk_id}" for doc in documents]
        
        # Generate embeddings
        print("⏳ Generating embeddings...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(documents)} documents to ChromaDB")
        return len(documents)
    
    def process_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> int:
        """
        Full pipeline: load, chunk, and index a file.
        
        Args:
            file_path: Path to file
            chunk_size: Chunk size for splitting
            overlap: Overlap between chunks
            
        Returns:
            Number of chunks indexed
        """
        # Load
        documents = self.load_file(file_path)
        
        # Chunk
        chunks = self.chunk_documents(documents, chunk_size, overlap)
        
        # Index
        return self.index_documents(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all documents from collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"🗑️ Cleared collection: {self.collection_name}")
