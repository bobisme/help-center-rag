"""Step to add context to document chunks."""

import asyncio
from typing import List, Tuple

from zenml import step
from zenml.logger import get_logger

from epic_rag.domain.models.document import Document, DocumentChunk
from epic_rag.infrastructure.container import container, setup_container

logger = get_logger(__name__)


@step
def add_document_context(
    doc_chunks: List[Tuple[Document, List[DocumentChunk]]],
    skip_enrichment: bool = False,
) -> List[Tuple[Document, List[DocumentChunk]]]:
    """Add contextual enrichment to document chunks.

    Args:
        doc_chunks: List of tuples containing the document and its chunks
        skip_enrichment: Whether to skip the enrichment process

    Returns:
        List of tuples containing the document and its enriched chunks
    """
    if skip_enrichment:
        logger.info("Skipping contextual enrichment as requested")
        return doc_chunks

    # Initialize container
    setup_container()

    # Get contextual enrichment service
    contextual_enrichment_service = container.get("contextual_enrichment_service")

    # Create an async function to process all documents
    async def process_all_documents():
        results = []

        for doc, chunks in doc_chunks:
            logger.info(f"Adding context to chunks for document: {doc.title}")

            # Apply contextual enrichment
            enriched_chunks = await contextual_enrichment_service.enrich_chunks(
                document=doc, chunks=chunks
            )

            results.append((doc, enriched_chunks))
            logger.info(
                f"Enriched {len(enriched_chunks)} chunks for document: {doc.title}"
            )

        return results

    # Run the async processing
    result = asyncio.run(process_all_documents())
    logger.info(f"Added context to chunks for {len(result)} documents")

    return result
