"""Application configuration settings."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DatabaseSettings:
    """Settings for SQLite database."""

    path: str = "data/epic_rag.db"
    enable_json: bool = True
    enable_foreign_keys: bool = True
    journal_mode: str = "WAL"  # Write-Ahead Logging for better concurrency
    synchronous: str = "NORMAL"  # Compromise between safety and performance
    temp_store: str = "MEMORY"  # Store temp tables and indices in memory


@dataclass
class QdrantSettings:
    """Settings for Qdrant vector database."""

    url: Optional[str] = None  # If None, use local instance
    api_key: Optional[str] = None
    collection_name: str = "epic_docs"
    vector_size: int = 1536  # Default for many embedding models
    distance: str = "Cosine"

    # Local settings (when url is None)
    local_path: str = "qdrant_data"


@dataclass
class EmbeddingSettings:
    """Settings for embedding service."""

    provider: str = "openai"  # openai, cohere, huggingface, etc.
    model: str = "text-embedding-3-small"  # For OpenAI
    dimensions: int = 1536
    batch_size: int = 20
    api_key: Optional[str] = None

    # HuggingFace specific settings
    device: str = "cpu"  # cpu, cuda, mps
    quantization: Optional[str] = None  # int8, fp16, etc.


@dataclass
class LLMSettings:
    """Settings for LLM service."""

    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.0  # Keep deterministic for retrieval
    max_tokens: int = 1024


@dataclass
class ChunkingSettings:
    """Settings for document chunking."""

    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    min_chunk_size: int = 300
    max_chunk_size: int = 800
    enable_dynamic_chunking: bool = True


@dataclass
class RetrievalSettings:
    """Settings for retrieval services."""

    first_stage_k: int = 20  # Number of chunks to retrieve in first stage
    second_stage_k: int = 5  # Number of chunks for second stage
    min_relevance_score: float = 0.7
    enable_query_transformation: bool = True
    enable_chunk_merging: bool = True
    max_merged_chunk_size: int = 1500


@dataclass
class Settings:
    """Application settings container."""

    # Core settings
    app_name: str = "Epic Documentation RAG"
    debug: bool = False
    environment: str = "development"  # development, testing, production

    # Data paths
    data_dir: str = "data"
    markdown_dir: str = "data/markdown"
    output_dir: str = "data/output"

    # Component settings
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load settings from environment variables."""
        settings = cls()

        # Core settings
        settings.app_name = os.getenv("EPIC_RAG_APP_NAME", settings.app_name)
        settings.debug = os.getenv("EPIC_RAG_DEBUG", "").lower() in ("true", "1", "yes")
        settings.environment = os.getenv("EPIC_RAG_ENVIRONMENT", settings.environment)

        # Database settings
        settings.database.path = os.getenv("EPIC_RAG_DB_PATH", settings.database.path)

        # Qdrant settings
        settings.qdrant.url = os.getenv("EPIC_RAG_QDRANT_URL")
        settings.qdrant.api_key = os.getenv("EPIC_RAG_QDRANT_API_KEY")
        settings.qdrant.collection_name = os.getenv(
            "EPIC_RAG_QDRANT_COLLECTION", settings.qdrant.collection_name
        )

        # Embedding settings
        settings.embedding.provider = os.getenv(
            "EPIC_RAG_EMBEDDING_PROVIDER", settings.embedding.provider
        )
        settings.embedding.model = os.getenv(
            "EPIC_RAG_EMBEDDING_MODEL", settings.embedding.model
        )
        settings.embedding.api_key = os.getenv("EPIC_RAG_EMBEDDING_API_KEY")

        # LLM settings
        settings.llm.provider = os.getenv(
            "EPIC_RAG_LLM_PROVIDER", settings.llm.provider
        )
        settings.llm.model = os.getenv("EPIC_RAG_LLM_MODEL", settings.llm.model)
        settings.llm.api_key = os.getenv("EPIC_RAG_LLM_API_KEY")

        return settings


# Default settings instance
settings = Settings.from_environment()
