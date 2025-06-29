"""Ollama LLM service implementation."""

import httpx
from typing import List, Dict, Any

from ...domain.services.llm_service import LLMService
from ..config.settings import LLMSettings

QUERY_TRANSFORM_PROMPT = """\
You are an expert in information retrieval and query optimization.
Your task is to rewrite the following user query to improve search results
for a help center documentation system.

The rewritten query should:
- Include at least 2-3 relevant keywords or terms related to the query
- Add domain-specific terminology that might be used in the documentation
- Maintain the original intent and meaning
- Be concise

Original query: {0}

IMPORTANT: ONLY return the rewritten query text. No explanations, no reasoning,
no extra text whatsoever.
Your entire response will be used directly as the search query.
"""

ANSWER_QUESTION_PROMPT = """\
You are an assistant for help documentation users. 
Your task is to answer questions based on the context provided from the help documentation.

CONTEXT:
{0}

USER QUESTION:
{1}

Guidelines for your answer:
1. Answer ONLY based on the information provided in the context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question accurately."
3. Use a professional, clear, and concise tone
4. Focus on providing practical, actionable information for users
5. Include specific steps or navigation paths when relevant
6. If you reference specific features, explain what they do
7. Do not make up information or guess about functionality
8. When appropriate, include any relevant cautions or warnings from the documentation

ANSWER:
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

        # Add max_tokens if provided in kwargs or settings
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        elif self._settings.max_tokens:
            payload["max_tokens"] = self._settings.max_tokens

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
        return await self.generate_text(
            QUERY_TRANSFORM_PROMPT.format(query), temperature=0.2
        )

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
        # Format context content into a single string
        if not context_chunks:
            context_text = "No relevant information found."
        else:
            context_chunks_formatted = []

            for i, chunk in enumerate(context_chunks):
                if chunk.get("content"):
                    title = chunk.get("title", "Unknown")
                    score = chunk.get("score", 0.0)
                    content = chunk.get("content", "")

                    # Format with relevance score to help the LLM prioritize information
                    chunk_text = f"[Document {i+1}] {title} (Relevance: {score:.2f})\n\n{content}"
                    context_chunks_formatted.append(chunk_text)

            context_text = (
                "\n\n" + "\n\n---\n\n".join(context_chunks_formatted) + "\n\n"
            )

            # As a fallback, if we somehow end up with empty context after filtering
            if not context_text or len(context_text.strip()) == 0:
                context_text = "No relevant information found."

        # Create the prompt with the context and question
        prompt = ANSWER_QUESTION_PROMPT.format(context_text, question)

        # Generate answer with slightly higher temperature for more natural responses
        temperature = kwargs.get("temperature", 0.3)
        return await self.generate_text(prompt, temperature=temperature)

    @property
    def model_name(self) -> str:
        """Get the name of the language model being used."""
        return self._settings.model
