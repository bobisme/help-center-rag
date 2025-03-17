"""Implementation of rank fusion service for combining search results."""

import time
import logging
from typing import Dict, List, Set, Tuple

from ...domain.models.document import DocumentChunk
from ...domain.models.retrieval import RetrievalResult
from ...domain.services.rank_fusion_service import RankFusionService

logger = logging.getLogger(__name__)


class RecipRankFusionService(RankFusionService):
    """Implementation of reciprocal rank fusion for combining search results."""

    def __init__(self, k: float = 60.0):
        """Initialize the reciprocal rank fusion service.

        Args:
            k: Constant in the RRF formula to mitigate impact of high rankings
               Higher values reduce the impact of top results
        """
        self.k = k

    async def fuse_results(
        self,
        vector_results: RetrievalResult,
        bm25_results: RetrievalResult,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> RetrievalResult:
        """Fuse results from BM25 and vector search using reciprocal rank fusion.

        This implementation uses a modified Reciprocal Rank Fusion algorithm
        with per-source weights to combine results from different search methods.

        Args:
            vector_results: Results from vector-based search
            bm25_results: Results from BM25 search
            bm25_weight: Weight to assign to BM25 results (0.0-1.0)
            vector_weight: Weight to assign to vector results (0.0-1.0)

        Returns:
            Combined retrieval result with fused relevance scores
        """
        # Start timing
        start_time = time.time()

        # Normalize weights to sum to 1.0
        total_weight = bm25_weight + vector_weight
        if total_weight != 1.0:
            bm25_weight = bm25_weight / total_weight
            vector_weight = vector_weight / total_weight

        # Build a combined document set from both result sets
        all_chunks: Dict[str, DocumentChunk] = {}
        chunk_scores: Dict[str, float] = {}

        # Helper function to add document with rank to the fused score
        def add_with_rank(chunks: List[DocumentChunk], weight: float) -> None:
            for rank, chunk in enumerate(chunks):
                if not chunk.id:
                    continue

                # Get or create chunk in the combined set
                if chunk.id not in all_chunks:
                    all_chunks[chunk.id] = chunk
                    chunk_scores[chunk.id] = 0.0

                # Add RRF score: weight * 1/(k + rank)
                rrf_score = weight * (1.0 / (self.k + rank + 1))
                chunk_scores[chunk.id] += rrf_score

        # Add documents from each source with appropriate weights
        add_with_rank(vector_results.chunks, vector_weight)
        add_with_rank(bm25_results.chunks, bm25_weight)

        # Sort by combined score
        sorted_chunks = sorted(
            [(chunk_id, score) for chunk_id, score in chunk_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Create result list with updated relevance scores
        result_chunks = []
        for chunk_id, score in sorted_chunks:
            chunk = all_chunks[chunk_id]
            # Update the relevance score to the fused score
            chunk.relevance_score = score
            result_chunks.append(chunk)

        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Taking the query ID from vector results
        query_id = vector_results.query_id

        # Calculate combined latency (sum of individual latencies + fusion time)
        # Make sure to handle None values for individual latencies
        vector_latency = vector_results.latency_ms or 0
        bm25_latency = bm25_results.latency_ms or 0
        combined_latency = vector_latency + bm25_latency + latency_ms

        # Create combined retrieval result
        result = RetrievalResult(
            query_id=query_id,
            chunks=result_chunks,
            latency_ms=combined_latency,
        )

        logger.info(
            f"Rank fusion combined {len(vector_results.chunks)} vector results and "
            f"{len(bm25_results.chunks)} BM25 results into {len(result_chunks)} results"
        )

        return result