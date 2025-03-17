"""Base class for cached embedding services."""

import logging
from typing import List, Optional, Any, Dict

from ...domain.models.document import DocumentChunk, EmbeddedChunk
from ...domain.models.retrieval import Query
from ...domain.services.embedding_service import EmbeddingService
from ...infrastructure.config.settings import Settings
from ...infrastructure.embedding.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class CachedEmbeddingService(EmbeddingService):
    """Base class that adds caching functionality to any embedding service."""

    def __init__(
        self,
        wrapped_service: EmbeddingService,
        settings: Settings,
        provider_name: str,
        model_name: str,
    ):
        """Initialize the cached embedding service.

        Args:
            wrapped_service: The embedding service to wrap with caching
            settings: Application settings
            provider_name: Name of the embedding provider (openai, huggingface, etc.)
            model_name: Name of the embedding model
        """
        self._wrapped = wrapped_service
        self._settings = settings
        self._provider = provider_name
        self._model = model_name

        # Initialize cache if enabled
        self._cache_enabled = settings.embedding.cache.enabled
        if self._cache_enabled:
            self._cache = EmbeddingCache(
                settings=settings,
                memory_cache_size=settings.embedding.cache.memory_size,
                cache_expiration_days=settings.embedding.cache.expiration_days,
            )

            # Clear old entries if configured
            if settings.embedding.cache.clear_on_startup:
                # Schedule clearing old entries without awaiting
                import asyncio

                asyncio.create_task(self._clear_old_entries())
        else:
            self._cache = None

        logger.info(
            f"Initialized cached embedding service for {provider_name}/{model_name} "
            f"(cache {'enabled' if self._cache_enabled else 'disabled'})"
        )

    async def _clear_old_entries(self):
        """Clear old entries from the cache."""
        if self._cache:
            count = await self._cache.clear_old_entries()
            logger.info(f"Cleared {count} old entries from embedding cache")

    async def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate an embedding vector for a text string.

        Checks the cache first, and falls back to the wrapped service if not found.

        Args:
            text: The text to embed
            is_query: Whether the text is a query or passage

        Returns:
            Vector embedding as a list of floats
        """
        if self._cache_enabled and self._cache:
            # Try to get from cache
            cached_embedding = await self._cache.get(
                text=text,
                provider=self._provider,
                model=self._model,
                is_query=is_query,
            )

            if cached_embedding is not None:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_embedding

        # Generate embedding with wrapped service
        embedding = await self._wrapped.embed_text(text, is_query)

        # Store in cache
        if self._cache_enabled and self._cache and embedding:
            await self._cache.set(
                text=text,
                embedding=embedding,
                provider=self._provider,
                model=self._model,
                is_query=is_query,
            )

        return embedding

    async def embed_query(self, query: Query) -> Query:
        """Generate an embedding for a query.

        Updates the query object with the embedding and returns it.

        Args:
            query: The query to embed

        Returns:
            Updated query with embedding
        """
        query.embedding = await self.embed_text(query.text, is_query=True)
        return query

    async def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Generate an embedding for a document chunk.

        Args:
            chunk: The document chunk to embed

        Returns:
            Embedded chunk with vector data
        """
        embedding = await self.embed_text(chunk.content, is_query=False)
        return EmbeddedChunk(
            id=chunk.id,
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=embedding,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            previous_chunk_id=chunk.previous_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
            relevance_score=chunk.relevance_score,
        )

    async def batch_embed_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[EmbeddedChunk]:
        """Generate embeddings for multiple chunks in a batch.

        Uses cache where possible for individual chunks.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of embedded chunks
        """
        # Process each chunk, potentially getting it from cache
        results = []
        for chunk in chunks:
            embedded_chunk = await self.embed_chunk(chunk)
            results.append(embedded_chunk)

        return results

    async def get_embedding_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate similarity between two embeddings.

        Delegates to the wrapped service.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        return await self._wrapped.get_embedding_similarity(embedding1, embedding2)

    @property
    def embedding_dimensions(self) -> int:
        """Get the dimensions of the embedding vectors."""
        return self._wrapped.embedding_dimensions

    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the cache.

        Returns:
            Dictionary of cache statistics or None if cache is disabled
        """
        if self._cache_enabled and self._cache:
            return await self._cache.get_stats()
        return None
