"""Step to embed and load chunks to the vector database."""

import asyncio
from typing import List, Tuple

from zenml import step
from zenml.logger import get_logger

from help_rag.domain.models.document import Document, DocumentChunk
from help_rag.infrastructure.container import container, setup_container

logger = get_logger(__name__)


@step
def load_to_vector_db(
    doc_chunks: List[Tuple[Document, List[DocumentChunk]]], dry_run: bool = False
) -> List[Tuple[Document, List[DocumentChunk], List[DocumentChunk]]]:
    """Embed chunks and load them to the vector database.

    Args:
        doc_chunks: List of tuples containing the document and its chunks
        dry_run: Whether to skip saving to the database

    Returns:
        List of tuples containing the document, its chunks, and embedded chunks
    """
    if dry_run:
        logger.info("Dry run - skipping vector database loading")
        empty_embedded = [(doc, chunks, []) for doc, chunks in doc_chunks]
        return empty_embedded

    # Initialize container
    setup_container()

    # Get embedding service and vector repository using type-based dependency injection
    from help_rag.domain.services.embedding_service import EmbeddingService
    from help_rag.domain.repositories.vector_repository import VectorRepository

    embedding_service = container[EmbeddingService]
    vector_repository = container[VectorRepository]

    # Create an async function to process all documents
    async def process_all_documents():
        results = []

        for doc, chunks in doc_chunks:
            logger.info(f"Embedding chunks for document: {doc.title}")

            # Generate embeddings for chunks
            embedded_chunks = await embedding_service.batch_embed_chunks(chunks)

            # Save to vector repository
            vector_ids = await vector_repository.batch_store_embeddings(embedded_chunks)

            # Update chunks with vector IDs
            for i, chunk in enumerate(embedded_chunks):
                chunk.vector_id = vector_ids[i]

            results.append((doc, chunks, embedded_chunks))
            logger.info(
                f"Embedded and stored {len(embedded_chunks)} chunks for document: {doc.title}"
            )

        return results

    # Run the async processing
    result = asyncio.run(process_all_documents())
    logger.info(f"Loaded chunks for {len(result)} documents to vector database")

    total_embedded = sum(len(embedded) for _, _, embedded in result)
    logger.info(f"Total embedded chunks: {total_embedded}")

    return result
