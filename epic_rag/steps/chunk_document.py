"""Step to chunk documents into smaller pieces."""

import asyncio
from typing import List, Tuple

from zenml import step
from zenml.logger import get_logger

from epic_rag.domain.models.document import Document, DocumentChunk
from epic_rag.infrastructure.container import container, setup_container

logger = get_logger(__name__)


@step
def chunk_document(
    documents: List[Document],
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
    dynamic_chunking: bool = True,
) -> List[Tuple[Document, List[DocumentChunk]]]:
    """Chunk documents into smaller pieces for processing.

    Args:
        documents: List of documents to chunk
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        dynamic_chunking: Whether to use dynamic chunking

    Returns:
        List of tuples containing the original document and its chunks
    """
    # Initialize container
    setup_container()

    # Get chunking service
    chunking_service = container.get("chunking_service")

    # Create an async function to process all documents
    async def process_all_documents():
        results = []

        for doc in documents:
            logger.info(f"Chunking document: {doc.title}")

            # Choose chunking method based on parameters
            if dynamic_chunking:
                chunks = await chunking_service.dynamic_chunk_document(
                    document=doc,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                )
            else:
                chunks = await chunking_service.chunk_document(
                    document=doc,
                    chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            results.append((doc, chunks))
            logger.info(f"Created {len(chunks)} chunks for document: {doc.title}")

        return results

    # Run the async processing
    result = asyncio.run(process_all_documents())
    logger.info(
        f"Chunked {len(result)} documents with a total of {sum(len(chunks) for _, chunks in result)} chunks"
    )

    return result
