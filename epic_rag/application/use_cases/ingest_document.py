"""Document ingestion use case."""

import time
from typing import Optional, Dict, Any

from ...domain.models.document import Document, DocumentChunk
from ...domain.repositories.document_repository import DocumentRepository
from ...domain.repositories.vector_repository import VectorRepository
from ...domain.services.chunking_service import ChunkingService
from ...domain.services.embedding_service import EmbeddingService
from ...domain.services.contextual_enrichment_service import ContextualEnrichmentService


class IngestDocumentUseCase:
    """Use case for ingesting documents into the system."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        contextual_enrichment_service: Optional[ContextualEnrichmentService] = None,
    ):
        """Initialize the use case.

        Args:
            document_repository: Repository for document storage
            vector_repository: Repository for vector operations
            chunking_service: Service for document chunking
            embedding_service: Service for generating embeddings
            contextual_enrichment_service: Optional service for enriching chunks with context
        """
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.contextual_enrichment_service = contextual_enrichment_service

    async def execute(
        self,
        document: Document,
        dynamic_chunking: bool = True,
        min_chunk_size: int = 300,
        max_chunk_size: int = 800,
        chunk_overlap: int = 50,
        extra_chunk_metadata: Optional[Dict[str, Any]] = None,
        apply_enrichment: bool = True,
        dry_run: bool = False,
    ) -> Document:
        """Execute the document ingestion process.

        Args:
            document: The document to ingest
            dynamic_chunking: Whether to use dynamic chunking
            min_chunk_size: Minimum chunk size when using dynamic chunking
            max_chunk_size: Maximum chunk size when using dynamic chunking
            chunk_overlap: Overlap between chunks
            extra_chunk_metadata: Additional metadata to add to chunks
            apply_enrichment: Whether to apply contextual enrichment to chunks before embedding
            dry_run: If True, processes the document but doesn't save to the database

        Returns:
            The processed document with chunks and embeddings
        """
        start_time = time.time()

        # Step 1: Save the document to the repository (skip if dry_run)
        if not dry_run:
            saved_document = await self.document_repository.save_document(document)
        else:
            # If dry_run, just use the document as is without saving
            saved_document = document
            # Ensure the document has an ID for reference purposes
            if not saved_document.id:
                saved_document.id = f"dry-run-{int(time.time())}"
            print("Dry run: Skipping document save to database")

        # Step 2: Chunk the document
        if dynamic_chunking:
            chunks = await self.chunking_service.dynamic_chunk_document(
                document=saved_document,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                metadata=extra_chunk_metadata,
            )
        else:
            chunks = await self.chunking_service.chunk_document(
                document=saved_document,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=extra_chunk_metadata,
            )

        # Step 3: Apply contextual enrichment if enabled and service is available
        if apply_enrichment and self.contextual_enrichment_service:
            print(f"Applying contextual enrichment to {len(chunks)} chunks...")
            enrichment_start = time.time()

            # Get the image description service from the contextual enrichment service if it's the enhanced version
            image_description_service = None
            # Check if it's an ImageEnhancedEnrichmentService by checking the class name
            if self.contextual_enrichment_service.__class__.__name__ == "ImageEnhancedEnrichmentService":
                # Use getattr to access the private attribute in a type-safe way
                image_description_service = getattr(
                    self.contextual_enrichment_service, "_image_description_service", None
                )

                # First, process images to get descriptions (this populates the cache in the service)
                enriched_chunks = (
                    await self.contextual_enrichment_service.enrich_chunks(
                        document=saved_document, chunks=chunks
                    )
                )

                # Now process each chunk to add image descriptions directly into the content
                if image_description_service is not None and hasattr(image_description_service, "process_chunk_images"):
                    print("Adding image descriptions directly to content...")
                    for i, chunk in enumerate(enriched_chunks):
                        # We need to replace the image descriptions at the top with descriptions under each image
                        # Step 1: Get the original context without image descriptions
                        context = chunk.metadata.get("context", "")

                        # Step 2: Create a new chunk with just the context (not image descriptions) at the top
                        # Use the original content (not the content with image descriptions at the top)
                        clean_chunk = DocumentChunk(
                            id=chunk.id,
                            document_id=chunk.document_id,
                            content=f"{context}\n\n{chunks[i].content}",  # Use original content
                            metadata=chunk.metadata,
                            embedding=chunk.embedding,
                            chunk_index=chunk.chunk_index,
                            previous_chunk_id=chunk.previous_chunk_id,
                            next_chunk_id=chunk.next_chunk_id,
                            relevance_score=chunk.relevance_score,
                        )

                        # Step 3: Process the chunk to add image descriptions under each image
                        processed_chunk = (
                            await image_description_service.process_chunk_images(
                                clean_chunk
                            )
                        )
                        enriched_chunks[i] = processed_chunk

                chunks = enriched_chunks
            else:
                # Regular contextual enrichment without image descriptions
                chunks = await self.contextual_enrichment_service.enrich_chunks(
                    document=saved_document, chunks=chunks
                )

            enrichment_time = (time.time() - enrichment_start) * 1000
            print(f"Contextual enrichment completed in {enrichment_time:.2f}ms")

        # Step 4: Save chunks to the repository (skip if dry_run)
        if not dry_run:
            for chunk in chunks:
                chunk.document_id = saved_document.id
                await self.document_repository.save_chunk(chunk)
        else:
            for chunk in chunks:
                chunk.document_id = saved_document.id
            print(f"Dry run: Skipping {len(chunks)} chunk saves to database")

        # Step 5: Generate embeddings for all chunks
        embedded_chunks = await self.embedding_service.batch_embed_chunks(chunks)

        # Step 6: Store embeddings in vector database (skip if dry_run)
        if not dry_run:
            vector_ids = await self.vector_repository.batch_store_embeddings(
                embedded_chunks
            )

            # Step 7: Update chunks with vector IDs (skip if dry_run)
            for i, chunk in enumerate(embedded_chunks):
                chunk.vector_id = vector_ids[i]
                await self.document_repository.save_chunk(chunk)
        else:
            # Assign dummy vector IDs for dry runs
            for i, chunk in enumerate(embedded_chunks):
                chunk.vector_id = f"dry-run-vector-{i}"
            print(f"Dry run: Skipping {len(embedded_chunks)} vector saves to database")

        # Update the document with chunks
        saved_document.chunks = chunks

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Document ingestion completed in {elapsed_ms:.2f}ms")

        return saved_document
