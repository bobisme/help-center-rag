"""LLM-based contextual enrichment service for document chunks."""

from typing import List, Protocol, runtime_checkable

from ..models.document import Document, DocumentChunk


@runtime_checkable
class ContextualEnrichmentService(Protocol):
    """Service for enriching document chunks with contextual information using LLMs."""

    async def enrich_chunk(
        self, document: Document, chunk: DocumentChunk
    ) -> DocumentChunk:
        """Enrich a document chunk with contextual information.

        This uses an LLM to generate a short contextual description that
        explains where the chunk fits within the overall document, which
        is then prepended to the chunk content.

        Args:
            document: The parent document containing the chunk
            chunk: The document chunk to enrich

        Returns:
            The enriched document chunk
        """
        ...

    async def enrich_chunks(
        self, document: Document, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Enrich multiple document chunks with contextual information.

        This is a bulk operation that may be more efficient than calling
        enrich_chunk individually for each chunk.

        Args:
            document: The parent document containing the chunks
            chunks: The document chunks to enrich

        Returns:
            The enriched document chunks
        """
        ...
