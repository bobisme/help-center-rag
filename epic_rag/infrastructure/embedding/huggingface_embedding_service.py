"""HuggingFace embedding service implementation."""

import asyncio
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModel

from ...domain.models.document import DocumentChunk, EmbeddedChunk
from ...domain.models.retrieval import Query
from ...domain.services.embedding_service import EmbeddingService
from ...infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingService(EmbeddingService):
    """Embedding service that uses local HuggingFace models."""

    def __init__(
        self,
        settings: Settings,
        model_name: str = "intfloat/e5-large-v2",
        dimensions: int = 1024,
        batch_size: int = 8,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """Initialize the HuggingFace embedding service.

        Args:
            settings: Application settings
            model_name: The HuggingFace model to use
            dimensions: Number of dimensions in the embedding
            batch_size: Max number of texts to embed in a single batch
            device: Device to run the model on (cuda, cpu, mps). If None, will use CUDA if available.
            max_length: Maximum token length for the model
        """
        self._settings = settings
        self._model_name = model_name
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_length = max_length

        # Determine device
        if device:
            self._device = device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        logger.info(f"Loading {model_name} on {self._device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)

        # Log GPU memory if using CUDA
        if self._device == "cuda":
            gpu_info = self._get_gpu_info()
            logger.info(f"GPU memory usage: {gpu_info}")

        logger.info(
            f"Initialized HuggingFace embedding service with model {model_name} on {self._device}"
        )

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {"status": "CUDA not available"}

        try:
            # Get current GPU device
            device = torch.cuda.current_device()

            # Get GPU name
            gpu_name = torch.cuda.get_device_name(device)

            # Get memory information
            total_memory = (
                torch.cuda.get_device_properties(device).total_memory / 1024**3
            )  # in GB
            reserved_memory = torch.cuda.memory_reserved(device) / 1024**3  # in GB
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # in GB
            free_memory = total_memory - allocated_memory

            return {
                "device": device,
                "name": gpu_name,
                "total_memory_gb": round(total_memory, 2),
                "reserved_memory_gb": round(reserved_memory, 2),
                "allocated_memory_gb": round(allocated_memory, 2),
                "free_memory_gb": round(free_memory, 2),
            }
        except Exception as e:
            return {"status": f"Error getting GPU info: {str(e)}"}

    def _average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Average pool the hidden states to get embeddings.

        Args:
            last_hidden_states: The last hidden states from the model
            attention_mask: The attention mask

        Returns:
            The pooled embeddings
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _prepare_text(self, text: str) -> str:
        """Prepare text for the model.

        For E5 models, each input text should start with "query: " or "passage: "
        For RAG purposes, we use "query: " for user queries and "passage: " for document chunks.

        Args:
            text: The text to prepare

        Returns:
            The prepared text
        """
        # Check if text already has a prefix
        if text.startswith("query: ") or text.startswith("passage: "):
            return text

        # Default to passage for document chunks
        return f"passage: {text}"

    def _embed_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            is_query: Whether the texts are queries or passages

        Returns:
            List of embedding vectors
        """
        # Prepare texts with the correct prefix
        prefix = "query: " if is_query else "passage: "
        prepared_texts = [
            (
                text
                if (text.startswith("query: ") or text.startswith("passage: "))
                else f"{prefix}{text}"
            )
            for text in texts
        ]

        # Tokenize inputs
        inputs = self._tokenizer(
            prepared_texts,
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._average_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Convert to list format with explicit type annotation
            # First convert to numpy array, then to Python list of lists
            numpy_array = embeddings.cpu().numpy()
            result: List[List[float]] = []
            
            for vec in numpy_array:
                float_list: List[float] = [float(x) for x in vec]
                result.append(float_list)
                
            return result

    async def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate an embedding vector for a text string.

        Args:
            text: The text to embed
            is_query: Whether the text is a query or passage

        Returns:
            Vector embedding as a list of floats
        """
        logger.debug(f"Embedding text of length {len(text)}")

        try:
            # Run in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            batch_result = await loop.run_in_executor(
                None, lambda: self._embed_batch([text], is_query=is_query)
            )
            result = batch_result[0] if batch_result else [0.0] * self._dimensions
            return result
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as a fallback
            return [0.0] * self._dimensions

    async def embed_query(self, query: Query) -> Query:
        """Generate an embedding for a query.

        Updates the query object with the embedding and returns it.

        Args:
            query: The query to embed

        Returns:
            Updated query with embedding
        """
        query.embedding = await self.embed_text(query.text, is_query=True)
        return query

    async def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Generate an embedding for a document chunk.

        Args:
            chunk: The document chunk to embed

        Returns:
            Embedded chunk with vector data
        """
        embedding = await self.embed_text(chunk.content, is_query=False)
        return EmbeddedChunk(
            id=chunk.id,
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=embedding,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            previous_chunk_id=chunk.previous_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
            relevance_score=chunk.relevance_score,
        )

    async def batch_embed_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[EmbeddedChunk]:
        """Generate embeddings for multiple chunks in a batch.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of embedded chunks
        """
        logger.info(f"Batch embedding {len(chunks)} chunks")

        # Process in batches
        results = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            logger.debug(
                f"Processing batch {i//self._batch_size + 1} of {(len(chunks)-1)//self._batch_size + 1}"
            )

            # Extract content for batch embedding
            texts = [chunk.content for chunk in batch]

            try:
                # Run in a thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, lambda: self._embed_batch(texts, is_query=False)
                )

                # Create embedded chunks with the results
                for j, chunk in enumerate(batch):
                    results.append(
                        EmbeddedChunk(
                            id=chunk.id,
                            content=chunk.content,
                            metadata=chunk.metadata,
                            embedding=embeddings[j],
                            document_id=chunk.document_id,
                            chunk_index=chunk.chunk_index,
                            previous_chunk_id=chunk.previous_chunk_id,
                            next_chunk_id=chunk.next_chunk_id,
                            relevance_score=chunk.relevance_score,
                        )
                    )
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Create fallback embeddings
                for chunk in batch:
                    results.append(
                        EmbeddedChunk(
                            id=chunk.id,
                            content=chunk.content,
                            metadata=chunk.metadata,
                            embedding=[0.0] * self._dimensions,
                            document_id=chunk.document_id,
                            chunk_index=chunk.chunk_index,
                            previous_chunk_id=chunk.previous_chunk_id,
                            next_chunk_id=chunk.next_chunk_id,
                            relevance_score=chunk.relevance_score,
                        )
                    )

        return results

    async def get_embedding_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @property
    def embedding_dimensions(self) -> int:
        """Get the dimensions of the embedding vectors."""
        return self._dimensions
