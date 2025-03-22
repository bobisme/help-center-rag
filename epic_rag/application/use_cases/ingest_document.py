"""Document ingestion use case."""

import asyncio
import time
from typing import List, Optional, Dict, Any

from ...domain.models.document import Document, DocumentChunk, EmbeddedChunk
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
            chunks = await self.contextual_enrichment_service.enrich_chunks(
                document=saved_document, 
                chunks=chunks
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
