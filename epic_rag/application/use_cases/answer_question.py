"""Question answering use case."""

import time
from typing import Dict, Any, Optional

from ...domain.models.retrieval import Query
from ...domain.services.embedding_service import EmbeddingService
from ...domain.services.llm_service import LLMService
from ...domain.services.retrieval_service import RetrievalService


class AnswerQuestionUseCase:
    """Use case for answering questions with RAG."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ):
        """Initialize the use case.

        Args:
            embedding_service: Service for generating embeddings
            retrieval_service: Service for retrieving documents
            llm_service: Service for generating answers
        """
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    async def execute(
        self,
        question: str,
        first_stage_k: int = 20,
        second_stage_k: int = 5,
        min_relevance_score: float = 0.3,  # Lowered from 0.7 to 0.3 for better recall
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_query_transformation: bool = True,
        merge_related_chunks: bool = True,
        temperature: float = 0.3,
        max_context_chunks: int = 5,
    ) -> Dict[str, Any]:
        """Execute the question answering process.

        Args:
            question: The user's question
            first_stage_k: Number of documents to retrieve in first stage
            second_stage_k: Number of documents to retrieve in second stage
            min_relevance_score: Minimum relevance score for filtering
            filter_metadata: Optional metadata filters
            use_query_transformation: Whether to transform the query
            merge_related_chunks: Whether to merge related chunks
            temperature: Temperature for LLM answer generation
            max_context_chunks: Maximum number of context chunks to include

        Returns:
            Dict containing the answer, context, and performance metrics
        """
        # Step 1: Start timing the full process
        start_time = time.time()

        # Step 2: Create and embed the query
        query = Query(text=question)
        embedded_query = await self.embedding_service.embed_query(query)

        # Step 3: Create the request for contextual retrieval
        from ...domain.models.retrieval import ContextualRetrievalRequest

        request = ContextualRetrievalRequest(
            query=embedded_query,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            min_relevance_score=min_relevance_score,
            filter_metadata=filter_metadata or {},
            use_query_context=use_query_transformation,
            merge_related_chunks=merge_related_chunks,
        )

        # Step 4: Execute the contextual retrieval
        retrieval_result = await self.retrieval_service.contextual_retrieval(request)
        retrieval_time = time.time() - start_time

        # Step 5: Prepare the context chunks for the LLM
        # Take only the top chunks based on relevance score
        chunks = retrieval_result.final_chunks[:max_context_chunks]

        # Check if we have any chunks at all
        if not chunks:
            # Log warning about no chunks found
            print(f"Warning: No chunks found for query: {question}")

        context_chunks = []
        for chunk in chunks:
            # Make sure the chunk has content before adding it
            if chunk.content and len(chunk.content.strip()) > 0:
                context_chunks.append(
                    {
                        "id": chunk.id,
                        "title": chunk.metadata.get("document_title", "Untitled"),
                        "content": chunk.content,
                        "score": getattr(
                            chunk, "score", getattr(chunk, "relevance_score", 0.0)
                        ),
                        "metadata": chunk.metadata,
                    }
                )

        # Step 6: Generate an answer using the LLM
        answer_start_time = time.time()

        # Debug logging for context chunks
        if context_chunks:
            print(f"Debug: Passing {len(context_chunks)} context chunks to LLM")
            for i, chunk in enumerate(context_chunks):
                print(
                    f"Debug: Context chunk {i+1} - Title: {chunk['title']}, Score: {chunk['score']:.4f}"
                )
                print(f"Debug: Content preview: {chunk['content'][:100]}...")
        else:
            print("Debug: No context chunks found to pass to LLM")

        answer = await self.llm_service.answer_question(
            question=question,
            context_chunks=context_chunks,
            temperature=temperature,
        )
        answer_time = time.time() - answer_start_time

        # Step 7: Calculate total time
        total_time = time.time() - start_time

        # Return the answer and context info
        return {
            "question": question,
            "answer": answer,
            "context_chunks": context_chunks,
            "metrics": {
                "total_time_ms": total_time * 1000,
                "retrieval_time_ms": retrieval_time * 1000,
                "answer_time_ms": answer_time * 1000,
                "chunks_retrieved": len(retrieval_result.final_chunks),
                "chunks_used": len(context_chunks),
                "model_used": self.llm_service.model_name,
            },
        }
