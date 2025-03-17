"""Retrieval service for implementing the Contextual Retrieval methodology."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models.document import DocumentChunk
from ..models.retrieval import (
    Query, 
    RetrievalResult,
    ContextualRetrievalRequest,
    ContextualRetrievalResult
)


class RetrievalService(ABC):
    """Service for retrieving relevant document chunks using contextual retrieval."""
    
    @abstractmethod
    async def retrieve(
        self,
        query: Query,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Basic vector retrieval of documents based on query similarity.
        
        Args:
            query: The query to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            Retrieval result with matching chunks
        """
        pass
    
    @abstractmethod
    async def transform_query(self, query: Query) -> Query:
        """Transform a query to better match the document corpus semantics.
        
        This is part of the Contextual Retrieval approach to improve initial
        retrieval by rewriting the query to better match the document language.
        
        Args:
            query: Original user query
            
        Returns:
            Transformed query with updated text
        """
        pass
    
    @abstractmethod
    async def filter_chunks_by_relevance(
        self,
        query: Query,
        chunks: List[DocumentChunk],
        min_score: float = 0.7
    ) -> List[DocumentChunk]:
        """Filter chunks by relevance score using LLM evaluation.
        
        Args:
            query: The user query
            chunks: List of chunks to filter
            min_score: Minimum relevance score threshold
            
        Returns:
            Filtered list of relevant chunks
        """
        pass
    
    @abstractmethod
    async def merge_related_chunks(
        self,
        chunks: List[DocumentChunk],
        max_merged_size: int = 1500
    ) -> List[DocumentChunk]:
        """Merge semantically related chunks for better context.
        
        Args:
            chunks: List of chunks to potentially merge
            max_merged_size: Maximum size of merged chunks
            
        Returns:
            List of merged chunks
        """
        pass
    
    @abstractmethod
    async def contextual_retrieval(
        self,
        request: ContextualRetrievalRequest
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
        pass