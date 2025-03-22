"""Ollama LLM service implementation."""

import httpx

from ...domain.services.llm_service import LLMService
from ..config.settings import LLMSettings

PROMPT = """\
You are an expert in information retrieval and query optimization.
Your task is to rewrite the following user query to improve search results
for the Applied Epic Agency Management System. Epic is a system used
by brokerages to manage accounting and associate financial reporting with
their books of business.

The rewritten query should:
- Include at least 2-3 relevant keywords or terms related to the query
- Add domain-specific terminology used in insurance and accounting
- Maintain the original intent and meaning
- Be concise

Original query: {0}

IMPORTANT: ONLY return the rewritten query text. No explanations, no reasoning,
no extra text whatsoever.
Your entire response will be used directly as the search query.
"""


class OllamaLLMService(LLMService):
    """Implementation of LLM service using local Ollama."""

    def __init__(self, settings: LLMSettings):
        """Initialize Ollama LLM service.

        Args:
            settings: LLM settings
        """
        self._settings = settings
        self._api_base = "http://localhost:11434"
        self._client = httpx.AsyncClient(timeout=60.0)

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API.

        Args:
            prompt: The prompt text
            **kwargs: Additional parameters for the model

        Returns:
            Generated text response
        """
        payload = {
            "model": self._settings.model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self._settings.temperature),
            "stream": False,
        }

        response = await self._client.post(
            f"{self._api_base}/api/generate", json=payload
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")

        result = response.json()
        return result.get("response", "")

    async def transform_query(self, query: str) -> str:
        """Transform a user query to better match document corpus.

        Uses a specialized prompt to rewrite the query for better retrieval.

        Args:
            query: Original user query

        Returns:
            Transformed query optimized for retrieval
        """

        return await self.generate_text(PROMPT.format(query), temperature=0.2)

    @property
    def model_name(self) -> str:
        """Get the name of the language model being used."""
        return self._settings.model
