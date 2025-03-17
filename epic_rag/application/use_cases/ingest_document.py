"""Document ingestion use case."""
import asyncio
import time
from typing import List, Optional, Dict, Any

from ...domain.models.document import Document, DocumentChunk, EmbeddedChunk
from ...domain.repositories.document_repository import DocumentRepository
from ...domain.repositories.vector_repository import VectorRepository
from ...domain.services.chunking_service import ChunkingService
from ...domain.services.embedding_service import EmbeddingService


class IngestDocumentUseCase:
    """Use case for ingesting documents into the system."""
    
    def __init__(
        self, 
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService
    ):
        """Initialize the use case.
        
        Args:
            document_repository: Repository for document storage
            vector_repository: Repository for vector operations
            chunking_service: Service for document chunking
            embedding_service: Service for generating embeddings
        """
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
    
    async def execute(
        self,
        document: Document,
        dynamic_chunking: bool = True,
        min_chunk_size: int = 300,
        max_chunk_size: int = 800,
        chunk_overlap: int = 50,
        extra_chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Execute the document ingestion process.
        
        Args:
            document: The document to ingest
            dynamic_chunking: Whether to use dynamic chunking
            min_chunk_size: Minimum chunk size when using dynamic chunking
            max_chunk_size: Maximum chunk size when using dynamic chunking
            chunk_overlap: Overlap between chunks
            extra_chunk_metadata: Additional metadata to add to chunks
            
        Returns:
            The processed document with chunks and embeddings
        """
        start_time = time.time()
        
        # Step 1: Save the document to the repository
        saved_document = await self.document_repository.save_document(document)
        
        # Step 2: Chunk the document
        if dynamic_chunking:
            chunks = await self.chunking_service.dynamic_chunk_document(
                document=saved_document,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                metadata=extra_chunk_metadata
            )
        else:
            chunks = await self.chunking_service.chunk_document(
                document=saved_document,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=extra_chunk_metadata
            )
        
        # Step 3: Save chunks to the repository
        for chunk in chunks:
            chunk.document_id = saved_document.id
            await self.document_repository.save_chunk(chunk)
        
        # Step 4: Generate embeddings for all chunks
        embedded_chunks = await self.embedding_service.batch_embed_chunks(chunks)
        
        # Step 5: Store embeddings in vector database
        vector_ids = await self.vector_repository.batch_store_embeddings(embedded_chunks)
        
        # Step 6: Update chunks with vector IDs
        for i, chunk in enumerate(embedded_chunks):
            chunk.vector_id = vector_ids[i]
            await self.document_repository.save_chunk(chunk)
        
        # Update the document with chunks
        saved_document.chunks = chunks
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Document ingestion completed in {elapsed_ms:.2f}ms")
        
        return saved_document