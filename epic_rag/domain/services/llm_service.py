"""LLM service for text generation and transformation."""

from typing import List, Dict, Any, Protocol, runtime_checkable


@runtime_checkable
class LLMService(Protocol):
    """Service for generating text and transformations using language models."""

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt.

        Args:
            prompt: The prompt text
            **kwargs: Additional parameters for model

        Returns:
            Generated text response
        """
        ...

    async def transform_query(self, query: str) -> str:
        """Transform a user query to better match document corpus.

        Args:
            query: Original user query

        Returns:
            Transformed query optimized for retrieval
        """
        ...

    async def answer_question(
        self, question: str, context_chunks: List[Dict[str, Any]], **kwargs
    ) -> str:
        """Generate an answer to a question based on the provided context chunks.

        Args:
            question: The user's question
            context_chunks: List of retrieved document chunks and their metadata
            **kwargs: Additional parameters for the model

        Returns:
            Generated answer based on the context
        """
        ...

    @property
    def model_name(self) -> str:
        """Get the name of the language model being used."""
        ...
