"""Vector repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..models.document import DocumentChunk, EmbeddedChunk
from ..models.retrieval import Query


class VectorRepository(ABC):
    """Interface for vector database operations."""

    @abstractmethod
    async def store_embedding(self, chunk: EmbeddedChunk) -> str:
        """Store a chunk embedding in the vector database.

        Returns:
            The vector ID from the database.
        """
        pass

    @abstractmethod
    async def delete_embedding(self, vector_id: str) -> bool:
        """Delete an embedding from the vector database."""
        pass

    @abstractmethod
    async def search_similar(
        self, query: Query, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Search for similar chunks based on vector similarity.

        Args:
            query: The query to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of document chunks with similarity scores
        """
        pass

    @abstractmethod
    async def batch_store_embeddings(self, chunks: List[EmbeddedChunk]) -> List[str]:
        """Store multiple embeddings at once.

        Returns:
            List of vector IDs from the database.
        """
        pass

    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        pass
