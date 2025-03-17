"""Implementation of lexical search using BM25 algorithm."""

import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
import re

from rank_bm25 import BM25Okapi
import numpy as np

from ...domain.models.document import DocumentChunk
from ...domain.models.retrieval import Query, RetrievalResult
from ...domain.repositories.document_repository import DocumentRepository
from ...domain.services.lexical_search_service import LexicalSearchService

logger = logging.getLogger(__name__)


class BM25SearchService(LexicalSearchService):
    """Implementation of lexical search using the BM25 algorithm."""

    def __init__(self, document_repository: DocumentRepository):
        """Initialize the BM25 search service.

        Args:
            document_repository: Repository for document storage
        """
        self.document_repository = document_repository
        self.bm25_index: Optional[BM25Okapi] = None
        self.document_lookup: Dict[int, str] = (
            {}
        )  # Maps BM25 index position to document ID
        self.tokenized_corpus: List[List[str]] = []
        self.indexed_documents: Set[str] = (
            set()
        )  # Track which document IDs have been indexed

        # Initialize the index asynchronously
        import asyncio

        try:
            # Create and run a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.reindex_all())
            loop.close()
            logger.info("BM25 index initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BM25 index: {e}")
            # If initialization fails, the index will be built on first use

    async def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase, remove punctuation and split into words
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    async def index_document(self, chunk: DocumentChunk) -> None:
        """Index a document chunk for BM25 search.

        Args:
            chunk: The document chunk to index
        """
        # Skip if already indexed
        if chunk.id in self.indexed_documents:
            return

        # Tokenize document
        tokenized_doc = await self._tokenize(chunk.content)

        # Add to corpus
        self.tokenized_corpus.append(tokenized_doc)
        self.document_lookup[len(self.tokenized_corpus) - 1] = chunk.id
        self.indexed_documents.add(chunk.id)

        # Rebuild index
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Indexed document chunk: {chunk.id}")

    async def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Index multiple document chunks for BM25 search.

        Args:
            chunks: The document chunks to index
        """
        new_docs = False

        for chunk in chunks:
            if chunk.id not in self.indexed_documents:
                # Tokenize document
                tokenized_doc = await self._tokenize(chunk.content)

                # Add to corpus
                self.tokenized_corpus.append(tokenized_doc)
                self.document_lookup[len(self.tokenized_corpus) - 1] = chunk.id
                self.indexed_documents.add(chunk.id)
                new_docs = True

        # Rebuild index if new documents were added
        if new_docs:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Indexed {len(chunks)} document chunks")

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
        # Start timing
        start_time = time.time()

        # Ensure we have an index
        if not self.bm25_index or not self.tokenized_corpus:
            logger.warning("BM25 index is empty. No results returned.")
            return RetrievalResult(
                query_id=query.id or str(uuid.uuid4()),
                chunks=[],
                latency_ms=0,
            )

        # Tokenize query
        tokenized_query = await self._tokenize(query.text)

        # Get BM25 scores for all documents
        doc_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k document indices
        top_k_indices = np.argsort(doc_scores)[::-1][:limit]

        # Convert to document IDs and scores
        results: List[Tuple[str, float]] = [
            (self.document_lookup[idx], doc_scores[idx])
            for idx in top_k_indices
            if doc_scores[idx] > 0  # Only include results with positive scores
        ]

        # Fetch full documents
        chunks = []
        for doc_id, score in results:
            chunk = await self.document_repository.get_chunk(doc_id)
            if chunk:
                # Add relevance score
                chunk.relevance_score = float(score)
                chunks.append(chunk)

                # Apply filters if specified
                if filters and chunk.metadata:
                    should_include = True
                    for key, value in filters.items():
                        if key not in chunk.metadata or chunk.metadata[key] != value:
                            should_include = False
                            break

                    if not should_include:
                        chunks.pop()  # Remove the chunk if it doesn't match filters

        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Create retrieval result
        result = RetrievalResult(
            query_id=query.id or str(uuid.uuid4()),
            chunks=chunks,
            latency_ms=latency_ms,
        )

        logger.info(
            f"BM25 search returned {len(chunks)} results with latency {latency_ms:.2f}ms"
        )
        return result

    async def reindex_all(self) -> None:
        """Rebuild the entire BM25 index from the document repository."""
        # Reset index data
        self.bm25_index = None
        self.document_lookup = {}
        self.tokenized_corpus = []
        self.indexed_documents = set()

        # Fetch all chunks from repository
        all_chunks = await self.document_repository.get_all_chunks()

        # Index all documents
        await self.index_documents(all_chunks)

        logger.info(f"Reindexed all documents: {len(all_chunks)} chunks in BM25 index")
