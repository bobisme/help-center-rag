"""Document repository interface."""

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

    async def find_document_by_source_page_id(
        self, source_page_id: str
    ) -> Optional[Document]:
        """Find a document by its source page ID."""
        ...

    async def replace_document(self, document: Document) -> Document:
        """Replace a document by its source page ID, deleting all existing chunks."""
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
        """Get all chunks from all documents.

        Args:
            limit: Maximum number of chunks to return

        Returns:
            List of all document chunks
        """
        ...

    async def find_orphaned_chunks(self) -> List[str]:
        """Find chunks that don't have a parent document.

        Returns:
            List of chunk IDs that are orphaned
        """
        ...

    async def delete_orphaned_chunks(self) -> int:
        """Delete chunks that don't have a parent document.

        Returns:
            Number of chunks deleted
        """
        ...

    async def vacuum_database(self) -> Tuple[float, float]:
        """Run vacuum operation on the database to reclaim unused space.

        Returns:
            Tuple of (size_before_mb, size_after_mb)
        """
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics.

        Returns:
            Dictionary of statistics
        """
        ...

    async def save_query(
        self,
        query_text: str,
        transformed_query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a query to the query history.

        Args:
            query_text: The original query text
            transformed_query: Optional transformed query text
            metadata: Optional metadata about the query (results, etc)

        Returns:
            The ID of the saved query
        """
        ...

    async def get_query_history(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get query history with optional date filtering.

        Args:
            limit: Maximum number of queries to return
            offset: Offset for pagination
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            List of query history records
        """
        ...
