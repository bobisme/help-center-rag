"""Implementation of contextual enrichment service using Ollama LLM."""

import asyncio
from typing import List

from ...domain.models.document import Document, DocumentChunk
from ...domain.services.contextual_enrichment_service import ContextualEnrichmentService
from ...domain.services.llm_service import LLMService


class OllamaContextualEnrichmentService(ContextualEnrichmentService):
    """Contextual enrichment service implemented using Ollama LLM."""

    def __init__(self, llm_service: LLMService):
        """Initialize the service with an LLM service.

        Args:
            llm_service: The LLM service to use for context generation
        """
        self._llm_service = llm_service

    async def enrich_chunk(
        self, document: Document, chunk: DocumentChunk
    ) -> DocumentChunk:
        """Enrich a document chunk with contextual information using Ollama.

        Args:
            document: The parent document containing the chunk
            chunk: The document chunk to enrich

        Returns:
            The enriched document chunk
        """
        # Create a prompt for the LLM based on Anthropic's example
        prompt = f"""<document> 
{document.content} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk.content} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Generate context using LLM
        context = await self._llm_service.generate_text(prompt, temperature=0.1)

        # Create a new chunk with the context prepended
        enriched_chunk = DocumentChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=f"{context}\n\n{chunk.content}",
            metadata={**chunk.metadata, "enriched": True, "context": context},
            embedding=chunk.embedding,
            chunk_index=chunk.chunk_index,
            previous_chunk_id=chunk.previous_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
            relevance_score=chunk.relevance_score,
        )

        return enriched_chunk

    async def enrich_chunks(
        self, document: Document, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Enrich multiple document chunks with contextual information.

        This processes chunks in parallel for better performance.

        Args:
            document: The parent document containing the chunks
            chunks: The document chunks to enrich

        Returns:
            The enriched document chunks
        """
        # Process chunks in parallel
        enrichment_tasks = [self.enrich_chunk(document, chunk) for chunk in chunks]

        enriched_chunks = await asyncio.gather(*enrichment_tasks)
        return list(enriched_chunks)
