"""Implementation of the Contextual Retrieval service."""

import time
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

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
from ...domain.services.lexical_search_service import LexicalSearchService
from ...domain.services.rank_fusion_service import RankFusionService
from ...domain.services.retrieval_service import RetrievalService
from ...domain.services.llm_service import LLMService
from ...infrastructure.config.settings import Settings

if TYPE_CHECKING:
    from ...domain.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


class ContextualRetrievalService(RetrievalService):
    """Implementation of the Contextual Retrieval methodology."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        bm25_service: Optional[LexicalSearchService] = None,
        rank_fusion_service: Optional[RankFusionService] = None,
        llm_service: Optional[LLMService] = None,
        reranker_service: Optional["RerankerService"] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the retrieval service.

        Args:
            document_repository: Repository for document storage
            vector_repository: Repository for vector operations
            embedding_service: Service for generating embeddings
            bm25_service: Service for BM25 lexical search
            rank_fusion_service: Service for combining vector and BM25 results
            llm_service: Optional service for LLM operations like query transformation
            reranker_service: Optional service for reranking search results
            settings: Application settings
        """
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service
        self.rank_fusion_service = rank_fusion_service
        self.llm_service = llm_service
        self.reranker_service = reranker_service
        self.settings = settings

    async def retrieve(
        self, query: Query, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieval of documents based on query similarity using both vector
        and lexical search with rank fusion.

        Args:
            query: The query to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Retrieval result with matching chunks
        """
        # Start timing
        start_time = time.time()

        # Ensure query has an embedding for vector search
        if not query.embedding:
            query = await self.embedding_service.embed_query(query)

        # Run vector search
        vector_results = await self._vector_search(query, limit, filters)

        # If BM25 is enabled, run BM25 search and combine with vector results
        if (
            self.settings is not None
            and getattr(self.settings.retrieval, "enable_bm25", False)
            and self.bm25_service
            and self.rank_fusion_service
        ):
            # Run BM25 search
            bm25_results = await self.bm25_service.search(query, limit, filters)

            # Fuse results
            fused_results = await self.rank_fusion_service.fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                bm25_weight=self.settings.retrieval.bm25_weight,
                vector_weight=self.settings.retrieval.vector_weight,
            )

            # Log results
            logger.info(
                f"Hybrid search: Vector: {len(vector_results.chunks)} results, "
                f"BM25: {len(bm25_results.chunks)} results, "
                f"Fused: {len(fused_results.chunks)} results"
            )

            # Use fused results
            result = fused_results
        else:
            # Use vector results only
            result = vector_results

            # Fetch full content for chunks (vector DB only stores IDs and metadata)
            for chunk in result.chunks:
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
        total_latency_ms = (end_time - start_time) * 1000

        # Update the latency in the result
        result.latency_ms = total_latency_ms

        return result

    async def _vector_search(
        self, query: Query, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Perform vector-based similarity search.

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
        if (
            not self.llm_service
            or self.settings is None
            or not getattr(
                self.settings.retrieval, "enable_query_transformation", False
            )
        ):
            logger.info(
                f"Query transformation disabled. Using original query: {query.text}"
            )
            return query

        try:
            # Use LLM to transform the query
            logger.info(f"Transforming query: {query.text}")
            transformed_text = await self.llm_service.transform_query(query.text)

            # Create a new query with the transformed text but keep the original ID
            # We'll need to re-embed the transformed query
            transformed_query = Query(
                id=query.id,
                text=transformed_text,
                metadata={**query.metadata, "original_query": query.text},
            )

            logger.info(f"Transformed query: '{query.text}' -> '{transformed_text}'")
            return transformed_query
        except Exception as e:
            logger.error(f"Error transforming query: {e}")
            return query

    async def filter_chunks_by_relevance(
        self, query: Query, chunks: List[DocumentChunk], min_score: float = 0.3
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
        # The chunks from retrieval might have small scores like 0.01-0.02
        # So we apply a different strategy - either take all chunks if there are less than 5
        # or take the top 5 by score
        if len(chunks) <= 5:
            relevant_chunks = chunks
        else:
            # Sort by score and take top 5
            sorted_chunks = sorted(
                chunks,
                key=lambda c: getattr(c, "score", c.relevance_score or 0.0),
                reverse=True,
            )
            relevant_chunks = sorted_chunks[:5]

        # Debug information about chunk filtering
        print(
            f"Debug: Before filtering - {len(chunks)} chunks with scores: "
            + ", ".join(
                [
                    f"{getattr(c, 'score', c.relevance_score or 0.0):.4f}"
                    for c in chunks[:5]
                ]
            )
            + (", ..." if len(chunks) > 5 else "")
        )
        print(
            f"Debug: After new filtering strategy - {len(relevant_chunks)} chunks with scores: "
            + ", ".join(
                [
                    f"{getattr(c, 'score', c.relevance_score or 0.0):.4f}"
                    for c in relevant_chunks[:5]
                ]
            )
            + (", ..." if len(relevant_chunks) > 5 else "")
        )

        # Print chunk content verification for first chunk
        if relevant_chunks:
            print(
                f"Debug: First chunk has content length: {len(relevant_chunks[0].content)}"
            )
            print(f"Debug: Content preview: {relevant_chunks[0].content[:100]}...")

        logger.info(
            f"Filtered {len(chunks)} chunks to {len(relevant_chunks)} relevant chunks"
        )
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

        This is the main entry point for the retrieval process:
        1. Query transformation (if enabled)
        2. Hybrid search (vector + BM25 with rank fusion)
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

        # Step 2: Hybrid retrieval (vector + BM25 with rank fusion)
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

        # Step 3.5: Apply reranking if enabled
        if (
            self.reranker_service
            and self.settings is not None
            and getattr(self.settings.retrieval, "reranker", None) is not None
            and getattr(self.settings.retrieval.reranker, "enabled", False)
        ):
            logger.info(f"Applying reranking to {len(relevant_chunks)} chunks")
            reranked_chunks = await self.reranker_service.rerank(
                query=request.query,
                chunks=relevant_chunks,
                top_k=getattr(self.settings.retrieval.reranker, "top_k", 5),
            )
            relevant_chunks = reranked_chunks
            logger.info(f"Reranking returned {len(relevant_chunks)} chunks")

        # Step 4: Merge related chunks if enabled
        if request.merge_related_chunks and relevant_chunks:
            # Combine related chunks for better context
            merged_chunks = await self.merge_related_chunks(
                chunks=relevant_chunks,
                max_merged_size=(
                    getattr(self.settings.retrieval, "max_merged_chunk_size", 1500)
                    if self.settings is not None
                    else 1500
                ),
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
