"""Implementation of the Contextual Retrieval service."""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

from ...domain.models.document import DocumentChunk
from ...domain.models.retrieval import (
    Query,
    RetrievalResult,
    ContextualRetrievalRequest,
    ContextualRetrievalResult,
)
from ...domain.repositories.document_repository import DocumentRepository
from ...domain.repositories.vector_repository import VectorRepository
from ...domain.services.embedding_service import EmbeddingService
from ...domain.services.retrieval_service import RetrievalService
from ...infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class ContextualRetrievalService(RetrievalService):
    """Implementation of the Contextual Retrieval methodology."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        settings: Settings,
    ):
        """Initialize the retrieval service.

        Args:
            document_repository: Repository for document storage
            vector_repository: Repository for vector operations
            embedding_service: Service for generating embeddings
            settings: Application settings
        """
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.settings = settings

    async def retrieve(
        self, query: Query, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Basic vector retrieval of documents based on query similarity.

        Args:
            query: The query to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Retrieval result with matching chunks
        """
        # Start timing
        start_time = time.time()

        # Ensure query has an embedding
        if not query.embedding:
            query = await self.embedding_service.embed_query(query)

        # Search for similar chunks
        similar_chunks = await self.vector_repository.search_similar(
            query=query, limit=limit, filters=filters
        )

        # Fetch full content for chunks (vector DB only stores IDs and metadata)
        for chunk in similar_chunks:
            # Get full chunk content from document repository
            full_chunk = await self.document_repository.get_chunk(chunk.id)
            if full_chunk:
                # Update the chunk content while keeping the relevance score
                chunk.content = full_chunk.content
                # Copy any missing metadata
                for key, value in full_chunk.metadata.items():
                    if key not in chunk.metadata:
                        chunk.metadata[key] = value

        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Create retrieval result
        result = RetrievalResult(
            query_id=query.id,
            chunks=similar_chunks,
            latency_ms=latency_ms,
        )

        return result

    async def transform_query(self, query: Query) -> Query:
        """Transform a query to better match the document corpus semantics.

        This is part of the Contextual Retrieval approach to improve initial
        retrieval by rewriting the query to better match the document language.

        Args:
            query: Original user query

        Returns:
            Transformed query with updated text
        """
        # TODO: Implement query transformation using LLM
        # For now, we'll return the original query
        logger.info(f"Query transformation not yet implemented. Using original query: {query.text}")
        return query

    async def filter_chunks_by_relevance(
        self, query: Query, chunks: List[DocumentChunk], min_score: float = 0.7
    ) -> List[DocumentChunk]:
        """Filter chunks by relevance score using vector similarity.

        In a production system, we would use an LLM to evaluate relevance.
        For this implementation, we'll use the vector similarity score.

        Args:
            query: The user query
            chunks: List of chunks to filter
            min_score: Minimum relevance score threshold

        Returns:
            Filtered list of relevant chunks
        """
        # Filter chunks based on relevance score
        relevant_chunks = [chunk for chunk in chunks if chunk.relevance_score and chunk.relevance_score >= min_score]
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(relevant_chunks)} relevant chunks")
        return relevant_chunks

    async def merge_related_chunks(
        self, chunks: List[DocumentChunk], max_merged_size: int = 1500
    ) -> List[DocumentChunk]:
        """Merge semantically related chunks for better context.

        For the initial implementation, we'll merge chunks from the same document
        that are adjacent to each other (based on chunk_index).

        Args:
            chunks: List of chunks to potentially merge
            max_merged_size: Maximum size of merged chunks

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
            
        # Group chunks by document ID
        doc_chunks: Dict[str, List[DocumentChunk]] = {}
        for chunk in chunks:
            if chunk.document_id not in doc_chunks:
                doc_chunks[chunk.document_id] = []
            doc_chunks[chunk.document_id].append(chunk)
        
        # Sort chunks by chunk_index within each document
        for doc_id in doc_chunks:
            doc_chunks[doc_id].sort(key=lambda c: c.chunk_index)
        
        merged_chunks = []
        
        # Process each document's chunks
        for doc_id, chunks in doc_chunks.items():
            current_merged = None
            
            for chunk in chunks:
                # Start a new merged chunk if we don't have one
                if current_merged is None:
                    current_merged = DocumentChunk(
                        id=f"merged_{chunk.id}",
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata={
                            **chunk.metadata,
                            "is_merged": True,
                            "merged_chunk_ids": [chunk.id],
                            "merged_chunk_count": 1,
                        },
                        chunk_index=chunk.chunk_index,
                        relevance_score=chunk.relevance_score,
                    )
                    continue
                
                # Check if this chunk is adjacent to the current merged chunk
                if chunk.chunk_index == current_merged.chunk_index + 1:
                    # Check if adding this chunk would exceed max size
                    combined_length = len(current_merged.content) + len(chunk.content)
                    if combined_length <= max_merged_size:
                        # Merge the chunks
                        current_merged.content += f"\n\n{chunk.content}"
                        current_merged.metadata["merged_chunk_ids"].append(chunk.id)
                        current_merged.metadata["merged_chunk_count"] += 1
                        # Update the relevance score to the maximum
                        if chunk.relevance_score and current_merged.relevance_score:
                            current_merged.relevance_score = max(
                                current_merged.relevance_score, chunk.relevance_score
                            )
                        continue
                
                # If we get here, we can't merge the current chunk
                # Add the current merged chunk to results and start a new one
                merged_chunks.append(current_merged)
                current_merged = DocumentChunk(
                    id=f"merged_{chunk.id}",
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "is_merged": True,
                        "merged_chunk_ids": [chunk.id],
                        "merged_chunk_count": 1,
                    },
                    chunk_index=chunk.chunk_index,
                    relevance_score=chunk.relevance_score,
                )
            
            # Add the last merged chunk
            if current_merged:
                merged_chunks.append(current_merged)
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks

    async def contextual_retrieval(
        self, request: ContextualRetrievalRequest
    ) -> ContextualRetrievalResult:
        """Perform full contextual retrieval following Anthropic's methodology.

        This is the main entry point for the two-stage retrieval process:
        1. Initial broad retrieval
        2. Query context-based focused retrieval
        3. Relevance filtering
        4. Context-aware merging

        Args:
            request: Contextual retrieval request with parameters

        Returns:
            Result with retrieved contexts and performance metrics
        """
        # Start timing
        start_time = time.time()
        
        # Initialize result
        result = ContextualRetrievalResult(
            query=request.query,
        )
        
        # Step 1: Transform query if enabled
        if request.use_query_context:
            request.query = await self.transform_query(request.query)
            result.query = request.query
        
        # Step 2: First-stage broad retrieval
        first_stage_results = await self.retrieve(
            query=request.query,
            limit=request.first_stage_k,
            filters=request.filter_metadata,
        )
        result.first_stage_results = first_stage_results
        
        # Record retrieval time
        retrieval_time = time.time()
        result.retrieval_latency_ms = (retrieval_time - start_time) * 1000
        
        # Step 3: Filter chunks by relevance
        relevant_chunks = await self.filter_chunks_by_relevance(
            query=request.query,
            chunks=first_stage_results.chunks,
            min_score=request.min_relevance_score,
        )
        
        # Step 4: Merge related chunks if enabled
        if request.merge_related_chunks and relevant_chunks:
            # Combine related chunks for better context
            merged_chunks = await self.merge_related_chunks(
                chunks=relevant_chunks,
                max_merged_size=self.settings.retrieval.max_merged_chunk_size,
            )
            result.final_chunks = merged_chunks
            
            # Create merged content by combining all chunks
            if merged_chunks:
                merged_content = "\n\n---\n\n".join(
                    [chunk.content for chunk in merged_chunks]
                )
                result.merged_content = merged_content
        else:
            # No merging, just use the filtered chunks
            result.final_chunks = relevant_chunks
            
            # Create merged content
            if relevant_chunks:
                merged_content = "\n\n---\n\n".join(
                    [chunk.content for chunk in relevant_chunks]
                )
                result.merged_content = merged_content
        
        # Calculate total time
        end_time = time.time()
        processing_time = end_time - retrieval_time
        total_time = end_time - start_time
        
        # Update timing metrics
        result.processing_latency_ms = processing_time * 1000
        result.total_latency_ms = total_time * 1000
        
        return result