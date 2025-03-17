"""Rank fusion service for combining multiple search result types."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from ..models.document import DocumentChunk
from ..models.retrieval import RetrievalResult


class RankFusionService(ABC):
    """Service for fusing search results from different retrieval methods."""

    @abstractmethod
    async def fuse_results(
        self,
        vector_results: RetrievalResult,
        bm25_results: RetrievalResult,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> RetrievalResult:
        """Fuse results from BM25 and vector search using reciprocal rank fusion.

        This method combines lexical (BM25) and semantic (vector) search results
        using a weighted approach that balances exact keyword matching with
        semantic similarity.

        Args:
            vector_results: Results from vector-based search
            bm25_results: Results from BM25 search
            bm25_weight: Weight to assign to BM25 results (0.0-1.0)
            vector_weight: Weight to assign to vector results (0.0-1.0)

        Returns:
            Combined retrieval result with fused relevance scores
        """
        pass
