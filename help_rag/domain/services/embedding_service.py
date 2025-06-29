"""Embedding service for vector representations."""

from typing import List, Protocol, runtime_checkable

from ..models.document import DocumentChunk, EmbeddedChunk
from ..models.retrieval import Query


@runtime_checkable
class EmbeddingService(Protocol):
    """Service for generating and managing vector embeddings."""

    async def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate an embedding vector for a text string.

        Args:
            text: The text to embed
            is_query: Whether the text is a query (True) or document (False)

        Returns:
            Vector embedding as a list of floats
        """
        ...

    async def embed_query(self, query: Query) -> Query:
        """Generate an embedding for a query.

        Updates the query object with the embedding and returns it.

        Args:
            query: The query to embed

        Returns:
            Updated query with embedding
        """
        ...

    async def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Generate an embedding for a document chunk.

        Args:
            chunk: The document chunk to embed

        Returns:
            Embedded chunk with vector data
        """
        ...

    async def batch_embed_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[EmbeddedChunk]:
        """Generate embeddings for multiple chunks in a batch.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of embedded chunks
        """
        ...

    async def get_embedding_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        ...

    @property
    def embedding_dimensions(self) -> int:
        """Get the dimensions of the embedding vectors."""
        ...
