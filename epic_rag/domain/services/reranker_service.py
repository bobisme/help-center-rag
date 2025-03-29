"""Reranker service interface for reranking search results."""

from typing import List, Optional, Protocol, runtime_checkable

from ..models.document import DocumentChunk
from ..models.retrieval import Query


@runtime_checkable
class RerankerService(Protocol):
    """Service interface for reranking retrieval results based on relevance to query."""

    async def rerank(
        self, query: Query, chunks: List[DocumentChunk], top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Rerank document chunks based on their relevance to the query.

        Args:
            query: The query to evaluate relevance against
            chunks: List of document chunks to rerank
            top_k: Optional limit on number of results to return

        Returns:
            Reranked list of document chunks with updated relevance scores
        """
        ...
