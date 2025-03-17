"""Implementation of reranker service using CrossEncoder models."""

import time
import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder

from ...domain.models.document import DocumentChunk
from ...domain.models.retrieval import Query
from ...domain.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


class CrossEncoderRerankerService(RerankerService):
    """Reranker implementation using Cross-Encoder models.

    Cross-Encoder models are specifically designed for reranking tasks
    and typically achieve higher accuracy than bi-encoders for relevance scoring.
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-large-v1"):
        """Initialize the cross-encoder reranker service.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        logger.info(
            f"Initializing CrossEncoderRerankerService with model: {model_name}"
        )
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Successfully loaded reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    async def rerank(
        self, query: Query, chunks: List[DocumentChunk], top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Rerank document chunks using the cross-encoder model.

        Args:
            query: The query to evaluate relevance against
            chunks: List of document chunks to rerank
            top_k: Optional limit on number of results to return

        Returns:
            Reranked list of document chunks with updated relevance scores
        """
        if not chunks:
            logger.info("No chunks to rerank")
            return []

        # Start timing
        start_time = time.time()

        # Prepare document texts
        documents = [chunk.content for chunk in chunks]

        # Get reranking scores from the model
        try:
            # Use the rank method which returns scores and reranked documents
            rerank_results = self.model.rank(
                query.text,
                documents,
                return_documents=True,
                top_k=top_k or len(documents),
            )

            # Create a mapping from content to index
            content_to_index = {chunk.content: i for i, chunk in enumerate(chunks)}

            # Create reranked list of chunks with updated scores
            reranked_chunks = []
            for result in rerank_results:
                score = result["score"]
                doc_text = result["text"]  # The API returns document text with this key

                if doc_text in content_to_index:
                    # Get the original chunk index
                    idx = content_to_index[doc_text]
                    chunk = chunks[idx]
                    # Create a new chunk with the updated score to avoid modifying the original
                    new_chunk = DocumentChunk(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        chunk_index=chunk.chunk_index,
                        relevance_score=float(
                            score
                        ),  # Update the relevance score with reranker score
                    )
                    reranked_chunks.append(new_chunk)

            # Calculate timing
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            logger.info(
                f"Reranked {len(chunks)} chunks to {len(reranked_chunks)} results in {latency_ms:.2f}ms"
            )

            return reranked_chunks

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fall back to original chunks on error
            return chunks
