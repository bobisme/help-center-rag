"""Lexical search service for implementing keyword-based document retrieval."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models.document import DocumentChunk
from ..models.retrieval import Query, RetrievalResult


class LexicalSearchService(ABC):
    """Service for indexing and retrieving documents using lexical search (BM25)."""

    @abstractmethod
    async def index_document(self, chunk: DocumentChunk) -> None:
        """Index a document chunk for BM25 search.

        Args:
            chunk: The document chunk to index
        """
        pass

    @abstractmethod
    async def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Index multiple document chunks for BM25 search.

        Args:
            chunks: The document chunks to index
        """
        pass

    @abstractmethod
    async def search(
        self, query: Query, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Search for documents using BM25 algorithm.

        Args:
            query: The query to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Retrieval result with matching chunks
        """
        pass

    @abstractmethod
    async def reindex_all(self) -> None:
        """Rebuild the entire BM25 index from the document repository."""
        pass