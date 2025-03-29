"""Faster BM25 implementation using huggingface's bm25s library."""

import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Set

import bm25s

from ...domain.models.document import DocumentChunk
from ...domain.models.retrieval import Query, RetrievalResult
from ...domain.repositories.document_repository import DocumentRepository
from ...domain.services.lexical_search_service import LexicalSearchService

logger = logging.getLogger(__name__)

# Set debug level for bm25s
import logging

logging.getLogger("bm25s").setLevel(logging.DEBUG)


class BM25SSearchService(LexicalSearchService):
    """Implementation of lexical search using huggingface's bm25s library.

    This implementation is significantly faster than the standard BM25Okapi implementation,
    particularly for larger document collections. The bm25s implementation uses several
    optimizations:

    1. Faster tokenization with a more efficient implementation
    2. Optimized indexing with sparse vector representations
    3. More efficient searching algorithm

    Design note:
    The huggingface bm25s library has a different API from rank_bm25:
    - It returns the actual document content strings as "indices" in its results
    - We need to map these back to our internal document IDs using corpus.index()
    - This is different from the rank_bm25 library which returns numeric indices
    """

    def __init__(self, document_repository: DocumentRepository):
        """Initialize the BM25S search service.

        Args:
            document_repository: Repository for document storage
        """
        self.document_repository = document_repository
        self.bm25_model = None
        self.document_lookup: Dict[int, str] = {}  # Maps index position to document ID
        self.corpus: List[str] = []  # Original document texts
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
            logger.info("BM25S index initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BM25S index: {e}")
            # If initialization fails, the index will be built on first use

    async def index_document(self, chunk: DocumentChunk) -> None:
        """Index a document chunk for BM25 search.

        Args:
            chunk: The document chunk to index
        """
        # Skip if already indexed
        if chunk.id in self.indexed_documents:
            return

        # Add to corpus
        self.corpus.append(chunk.content)
        self.document_lookup[len(self.corpus) - 1] = chunk.id
        self.indexed_documents.add(chunk.id)

        # Rebuild index
        self._rebuild_index()

        logger.info(f"Indexed document chunk: {chunk.id}")

    async def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Index multiple document chunks for BM25 search.

        Args:
            chunks: The document chunks to index
        """
        new_docs = False

        for chunk in chunks:
            if chunk.id not in self.indexed_documents:
                # Add to corpus
                self.corpus.append(chunk.content)
                self.document_lookup[len(self.corpus) - 1] = chunk.id
                self.indexed_documents.add(chunk.id)
                new_docs = True

        # Rebuild index if new documents were added
        if new_docs:
            self._rebuild_index()

        logger.info(f"Indexed {len(chunks)} document chunks")

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index with the current corpus."""
        if not self.corpus:
            logger.warning("Cannot build index: corpus is empty")
            return

        # Log what we're indexing
        logger.info(f"Building BM25S index with {len(self.corpus)} documents")

        # Create the BM25 model
        self.bm25_model = bm25s.BM25(corpus=self.corpus)

        # Tokenize and index the corpus
        logger.info("Tokenizing corpus...")
        tokenized_corpus = bm25s.tokenize(self.corpus)
        logger.info(f"Tokenized corpus with {len(tokenized_corpus)} documents")

        # Debug first document tokenization
        if len(tokenized_corpus) > 0:
            logger.info(f"First document tokens sample: {tokenized_corpus[0][:20]}")

        # Note: The huggingface bm25s library uses the original corpus documents
        # as the indices in its results, not numeric indices. This is different
        # from the rank_bm25 library. We need to adapt our lookup mechanism.

        # Index the tokenized corpus
        logger.info("Indexing tokenized corpus...")
        self.bm25_model.index(tokenized_corpus)
        logger.info("Indexing complete")

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
        if not self.bm25_model or not self.corpus:
            logger.warning("BM25S index is empty. No results returned.")
            return RetrievalResult(
                query_id=query.id or str(uuid.uuid4()),
                chunks=[],
                latency_ms=0,
            )

        # Tokenize the query
        logger.info(f"Original query text: '{query.text}'")
        tokenized_query = bm25s.tokenize([query.text])
        logger.info(f"Tokenized query: {tokenized_query}")

        # Log debug info about the corpus and token statistics
        logger.info(f"Corpus size: {len(self.corpus)}")
        logger.info(f"First document sample: '{self.corpus[0][:100]}...'")

        # Ensure k doesn't exceed corpus size
        adjusted_limit = min(limit, len(self.corpus))
        if adjusted_limit < limit:
            logger.warning(
                f"Reduced search limit from {limit} to {adjusted_limit} due to corpus size"
            )

        # Perform the search to get top-k results
        doc_indices, scores = self.bm25_model.retrieve(
            tokenized_query, k=adjusted_limit
        )
        logger.info(f"Retrieved doc indices: {doc_indices}")
        logger.info(f"Retrieved scores: {scores}")

        # Extract results for the first (and only) query
        doc_indices = doc_indices[0]
        scores = scores[0]

        logger.info(f"Doc indices count: {len(doc_indices)}")
        logger.info(f"Scores count: {len(scores)}")

        # In the bm25s library, doc_indices contains the actual document content
        # We need to map this back to our document IDs
        results = []
        for i, (doc_content, score) in enumerate(zip(doc_indices, scores)):
            # Find the index in our corpus that matches this content
            try:
                corpus_idx = self.corpus.index(doc_content)
                if corpus_idx in self.document_lookup:
                    doc_id = self.document_lookup[corpus_idx]
                    # Only include results with positive scores
                    if score > 0:
                        results.append((doc_id, score))
                        logger.info(
                            f"Including result: corpus_idx={corpus_idx}, doc_id={doc_id}, score={score}"
                        )
                    else:
                        logger.info(
                            f"Skipping result with zero score: doc_id={doc_id}, score={score}"
                        )
                else:
                    logger.warning(f"Corpus index {corpus_idx} not found in lookup")
            except ValueError:
                # This could happen if the document content from bm25s doesn't match exactly
                logger.warning(
                    f"Document content not found in corpus: {doc_content[:50]}..."
                )

        logger.info(f"Mapped {len(results)} results with positive scores")

        # Fetch full documents
        chunks = []
        for doc_id, score in results:
            chunk = await self.document_repository.get_chunk(doc_id)
            if chunk:
                # Add relevance score
                chunk.relevance_score = float(score)
                logger.info(f"Retrieved chunk: id={chunk.id}, score={score}")

                # Apply filters if specified
                should_include = True
                if filters and chunk.metadata:
                    for key, value in filters.items():
                        if key not in chunk.metadata or chunk.metadata[key] != value:
                            should_include = False
                            logger.info(
                                f"Filtering out chunk {chunk.id} due to metadata mismatch"
                            )
                            break

                if should_include:
                    chunks.append(chunk)
                    logger.info(f"Added chunk {chunk.id} to results")
            else:
                logger.warning(f"Could not find chunk with ID {doc_id}")

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
            f"BM25S search returned {len(chunks)} results with latency {latency_ms:.2f}ms"
        )
        return result

    async def reindex_all(self) -> None:
        """Rebuild the entire BM25 index from the document repository."""
        # Reset index data
        self.bm25_model = None
        self.document_lookup = {}
        self.corpus = []
        self.indexed_documents = set()

        # Fetch all chunks from repository
        all_chunks = await self.document_repository.get_all_chunks()
        
        # Convert to list to ensure proper length calculation
        all_chunks_list = list(all_chunks)

        # Index all documents
        await self.index_documents(all_chunks_list)

        logger.info(f"Reindexed all documents: {len(all_chunks_list)} chunks in BM25S index")
