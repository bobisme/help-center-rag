"""OpenAI embedding service implementation."""

import logging
import numpy as np
from typing import List

from openai import AsyncOpenAI

from ...domain.models.document import DocumentChunk, EmbeddedChunk
from ...domain.models.retrieval import Query
from ...domain.services.embedding_service import EmbeddingService
from ...infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service that uses OpenAI's embedding models."""

    def __init__(
        self,
        settings: Settings,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
    ):
        """Initialize the OpenAI embedding service.

        Args:
            settings: Application settings
            model: The OpenAI embedding model to use
            dimensions: Number of dimensions in the embedding
            batch_size: Max number of texts to embed in a single API call
        """
        self._settings = settings
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info(f"Initialized OpenAI embedding service with model {model}")

    async def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a text string.

        Args:
            text: The text to embed

        Returns:
            Vector embedding as a list of floats
        """
        logger.debug(f"Embedding text of length {len(text)}")

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
                dimensions=self._dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as a fallback
            return [0.0] * self._dimensions

    async def embed_query(self, query: Query) -> Query:
        """Generate an embedding for a query.

        Updates the query object with the embedding and returns it.

        Args:
            query: The query to embed

        Returns:
            Updated query with embedding
        """
        query.embedding = await self.embed_text(query.text)
        return query

    async def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Generate an embedding for a document chunk.

        Args:
            chunk: The document chunk to embed

        Returns:
            Embedded chunk with vector data
        """
        embedding = await self.embed_text(chunk.content)
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

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of embedded chunks
        """
        logger.info(f"Batch embedding {len(chunks)} chunks")

        # Process in batches to avoid API limits
        results = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            logger.debug(
                f"Processing batch {i//self._batch_size + 1} of {(len(chunks)-1)//self._batch_size + 1}"
            )

            # Extract content for batch embedding
            texts = [chunk.content for chunk in batch]

            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                )

                # Create embedded chunks with the results
                for j, chunk in enumerate(batch):
                    embedding = response.data[j].embedding
                    results.append(
                        EmbeddedChunk(
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
                    )
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Create fallback embeddings
                for chunk in batch:
                    results.append(
                        EmbeddedChunk(
                            id=chunk.id,
                            content=chunk.content,
                            metadata=chunk.metadata,
                            embedding=[0.0] * self._dimensions,
                            document_id=chunk.document_id,
                            chunk_index=chunk.chunk_index,
                            previous_chunk_id=chunk.previous_chunk_id,
                            next_chunk_id=chunk.next_chunk_id,
                            relevance_score=chunk.relevance_score,
                        )
                    )

        return results

    async def get_embedding_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @property
    def embedding_dimensions(self) -> int:
        """Get the dimensions of the embedding vectors."""
        return self._dimensions
