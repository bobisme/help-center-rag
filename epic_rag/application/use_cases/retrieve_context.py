"""Context retrieval use case."""
import time
from typing import List, Optional, Dict, Any

from ...domain.models.retrieval import (
    Query, 
    ContextualRetrievalRequest,
    ContextualRetrievalResult
)
from ...domain.services.embedding_service import EmbeddingService
from ...domain.services.retrieval_service import RetrievalService


class RetrieveContextUseCase:
    """Use case for retrieving context using the Contextual Retrieval methodology."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService
    ):
        """Initialize the use case.
        
        Args:
            embedding_service: Service for generating embeddings
            retrieval_service: Service for retrieving documents
        """
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
    
    async def execute(
        self,
        query_text: str,
        first_stage_k: int = 20,
        second_stage_k: int = 5,
        min_relevance_score: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_query_transformation: bool = True,
        merge_related_chunks: bool = True
    ) -> ContextualRetrievalResult:
        """Execute the contextual retrieval process.
        
        Args:
            query_text: The user's query text
            first_stage_k: Number of documents to retrieve in first stage
            second_stage_k: Number of documents to retrieve in second stage
            min_relevance_score: Minimum relevance score for filtering
            filter_metadata: Optional metadata filters
            use_query_transformation: Whether to transform the query
            merge_related_chunks: Whether to merge related chunks
            
        Returns:
            The retrieval result with context
        """
        start_time = time.time()
        
        # Step 1: Create and embed the query
        query = Query(text=query_text)
        embedded_query = await self.embedding_service.embed_query(query)
        
        # Step 2: Create the request
        request = ContextualRetrievalRequest(
            query=embedded_query,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            min_relevance_score=min_relevance_score,
            filter_metadata=filter_metadata or {},
            use_query_context=use_query_transformation,
            merge_related_chunks=merge_related_chunks
        )
        
        # Step 3: Execute the contextual retrieval
        result = await self.retrieval_service.contextual_retrieval(request)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Add timing information
        if result.total_latency_ms is None:
            result.total_latency_ms = elapsed_ms
            
        return result