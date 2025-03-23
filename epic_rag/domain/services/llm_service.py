"""LLM service for text generation and transformation."""

from abc import ABC, abstractmethod


class LLMService(ABC):
    """Service for generating text and transformations using language models."""

    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt.

        Args:
            prompt: The prompt text
            **kwargs: Additional parameters for model

        Returns:
            Generated text response
        """

    @abstractmethod
    async def transform_query(self, query: str) -> str:
        """Transform a user query to better match document corpus.

        Args:
            query: Original user query

        Returns:
            Transformed query optimized for retrieval
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the language model being used."""
