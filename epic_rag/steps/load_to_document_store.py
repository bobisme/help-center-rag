"""Step to load documents and chunks to the document store."""

import asyncio
from typing import List, Tuple

from zenml import step
from zenml.logger import get_logger

from epic_rag.domain.models.document import Document, DocumentChunk
from epic_rag.infrastructure.container import container, setup_container

logger = get_logger(__name__)


@step
def load_to_document_store(
    doc_chunks: List[Tuple[Document, List[DocumentChunk]]], dry_run: bool = False
) -> List[Tuple[Document, List[DocumentChunk]]]:
    """Load documents and chunks to the document store.

    Args:
        doc_chunks: List of tuples containing the document and its chunks
        dry_run: Whether to skip saving to the database

    Returns:
        The same list of document-chunks tuples, for passing to the next step
    """
    if dry_run:
        logger.info("Dry run - skipping document store loading")
        return doc_chunks

    # Initialize container
    setup_container()

    # Get document repository using type-based dependency injection
    from epic_rag.domain.repositories.document_repository import DocumentRepository

    document_repository = container[DocumentRepository]

    # Create an async function to process all documents
    async def process_all_documents():
        for doc, chunks in doc_chunks:
            logger.info(f"Storing document in repository: {doc.title}")

            # Save document
            saved_doc = await document_repository.save_document(doc)

            # Save chunks
            for chunk in chunks:
                chunk.document_id = saved_doc.id
                await document_repository.save_chunk(chunk)

            logger.info(f"Stored document {doc.title} with {len(chunks)} chunks")

        return doc_chunks

    # Run the async processing
    result = asyncio.run(process_all_documents())
    logger.info(f"Loaded {len(result)} documents to document store")

    return result
