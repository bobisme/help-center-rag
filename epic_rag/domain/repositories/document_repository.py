"""Document repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..models.document import Document, DocumentChunk


class DocumentRepository(ABC):
    """Interface for document storage and retrieval."""

    @abstractmethod
    async def save_document(self, document: Document) -> Document:
        """Save a document to the repository."""
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """List documents with optional filtering."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Update an existing document."""
        pass

    @abstractmethod
    async def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save a document chunk."""
        pass

    @abstractmethod
    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        pass

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        pass
        
    @abstractmethod
    async def get_all_chunks(self, limit: int = 10000) -> List[DocumentChunk]:
        """Get all chunks from all documents.
        
        Args:
            limit: Maximum number of chunks to return
            
        Returns:
            List of all document chunks
        """
        pass
        
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics.
        
        Returns:
            Dictionary of statistics
        """
        pass
