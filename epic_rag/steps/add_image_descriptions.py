"""Step to add image descriptions to document chunks."""

import asyncio
import re
from typing import List, Tuple

from zenml import step
from zenml.logger import get_logger

from epic_rag.domain.models.document import Document, DocumentChunk
from epic_rag.infrastructure.container import container, setup_container

logger = get_logger(__name__)


@step
def add_image_descriptions(
    doc_chunks: List[Tuple[Document, List[DocumentChunk]]],
    skip_image_descriptions: bool = False,
) -> List[Tuple[Document, List[DocumentChunk]]]:
    """Add descriptions to images in document chunks.

    Args:
        doc_chunks: List of tuples containing the document and its chunks
        skip_image_descriptions: Whether to skip adding image descriptions

    Returns:
        List of tuples containing the document and its chunks with image descriptions
    """
    if skip_image_descriptions:
        logger.info("Skipping image description generation as requested")
        return doc_chunks

    # Initialize container
    setup_container()

    # Get image description service
    image_description_service = container.get("image_description_service")

    # Check if service has the required method
    if not hasattr(image_description_service, "process_chunk_images"):
        logger.warning(
            "Image description service does not support processing chunk images. Skipping."
        )
        return doc_chunks

    # Create an async function to process all documents
    async def process_all_documents():
        results = []
        total_images = 0

        for doc, chunks in doc_chunks:
            logger.info(
                f"Adding image descriptions to chunks for document: {doc.title}"
            )
            enriched_chunks = []
            doc_total_images = 0

            for chunk in chunks:
                # Count images in the chunk - handle both ![](image.png) and ![alt](image.png) format
                image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
                matches = list(re.finditer(image_pattern, chunk.content))
                doc_total_images += len(matches)
                logger.debug(f"Found {len(matches)} images in chunk {chunk.chunk_index}")

                # Only process if the chunk has images
                if matches:
                    # Process image descriptions
                    updated_chunk = (
                        await image_description_service.process_chunk_images(chunk)
                    )
                    enriched_chunks.append(updated_chunk)
                else:
                    # No images to process
                    enriched_chunks.append(chunk)

            total_images += doc_total_images
            results.append((doc, enriched_chunks))
            logger.info(
                f"Added descriptions for {doc_total_images} images across {len(chunks)} chunks for document: {doc.title}"
            )

        logger.info(f"Total images processed: {total_images}")
        return results

    # Run the async processing
    result = asyncio.run(process_all_documents())
    logger.info(f"Added image descriptions to chunks for {len(result)} documents")

    return result
