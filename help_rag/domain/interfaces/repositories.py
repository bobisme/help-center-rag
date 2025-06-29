"""Repository interfaces for the system."""

from typing import List, Optional, Dict, Any, Tuple, Protocol, runtime_checkable
from datetime import datetime

from ..models.document import Document, DocumentChunk


@runtime_checkable
class DocumentRepository(Protocol):
    """Interface for document storage and retrieval."""

    async def save_document(self, document: Document) -> Document:
        """Save a document to the repository."""
        ...

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        ...

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """List documents with optional filtering."""
        ...

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        ...

    async def update_document(self, document: Document) -> Document:
        """Update an existing document."""
        ...

    async def find_document_by_epic_page_id(
        self, epic_page_id: str
    ) -> Optional[Document]:
        """Find a document by its Epic page ID."""
        ...

    async def replace_document(self, document: Document) -> Document:
        """Replace a document by its Epic page ID, deleting all existing chunks."""
        ...

    async def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save a document chunk."""
        ...

    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        ...

    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        ...

    async def get_all_chunks(self, limit: int = 10000) -> List[DocumentChunk]:
        """Get all chunks from all documents."""
        ...

    async def find_orphaned_chunks(self) -> List[str]:
        """Find chunks that don't have a parent document."""
        ...

    async def delete_orphaned_chunks(self) -> int:
        """Delete chunks that don't have a parent document."""
        ...

    async def vacuum_database(self) -> Tuple[float, float]:
        """Run vacuum operation on the database to reclaim unused space."""
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        ...

    async def save_query(
        self,
        query_text: str,
        transformed_query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a query to the query history."""
        ...

    async def get_query_history(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get query history with optional date filtering."""
        ...


@runtime_checkable
class VectorRepository(Protocol):
    """Interface for vector database operations."""

    async def store_embedding(
        self,
        chunk_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an embedding in the vector database."""
        ...

    async def batch_store_embeddings(self, chunks: List[DocumentChunk]) -> List[str]:
        """Store multiple embeddings in the vector database."""
        ...

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        ...

    async def delete_embedding(self, vector_id: str) -> bool:
        """Delete an embedding by ID."""
        ...

    async def delete_embeddings(self, vector_ids: List[str]) -> int:
        """Delete multiple embeddings by IDs."""
        ...

    async def delete_embeddings_by_document_id(self, document_id: str) -> int:
        """Delete all embeddings associated with a document."""
        ...

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector collection."""
        ...
