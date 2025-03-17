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
    vector_size: int = 1024  # Match E5-large-v2 dimensions
    distance: str = "Cosine"

    # Local settings (when url is None)
    local_path: str = "qdrant_data"


@dataclass
class CacheSettings:
    """Settings for embedding cache."""
    
    enabled: bool = True
    memory_size: int = 1000  # Number of entries to keep in memory
    expiration_days: int = 30  # Number of days to keep entries
    clear_on_startup: bool = False  # Whether to clear expired entries on startup


@dataclass
class EmbeddingSettings:
    """Settings for embedding service."""

    provider: str = "huggingface"  # huggingface, openai, gemini, etc.
    dimensions: int = 1536  # Default dimensions
    batch_size: int = 20
    api_key: Optional[str] = None

    # Provider-specific model names
    openai_model: str = "text-embedding-3-small"
    gemini_model: str = "gemini-embedding-exp-03-07"
    huggingface_model: str = "intfloat/e5-large-v2"

    # HuggingFace specific settings
    device: str = "cuda"  # cuda, cpu, mps
    max_length: int = 512  # Maximum token length
    quantization: Optional[str] = None  # int8, fp16, etc.

    # Model-specific dimensions
    openai_dimensions: int = 1536  # text-embedding-3-small dimensions
    gemini_dimensions: int = 768  # gemini-embedding dimensions
    huggingface_dimensions: int = 1024  # e5-large-v2 dimensions
    
    # Cache settings
    cache: CacheSettings = field(default_factory=CacheSettings)

    @property
    def model(self):
        if self.provider == "openai":
            return self.openai_model
        elif self.provider == "gemini":
            return self.gemini_model
        elif self.provider == "huggingface":
            return self.huggingface_model
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


@dataclass
class LLMSettings:
    """Settings for LLM service."""

    provider: str = "ollama"
    model: str = "gemma3:27b"  # Using more powerful model for better transformations
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

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key.

        Returns the embedding API key if provider is OpenAI,
        otherwise falls back to the LLM API key if provider is OpenAI,
        or returns None if neither is available.
        """
        if self.embedding.provider.lower() == "openai" and self.embedding.api_key:
            return self.embedding.api_key
        elif self.llm.provider.lower() == "openai" and self.llm.api_key:
            return self.llm.api_key
        return os.getenv("OPENAI_API_KEY")

    @property
    def gemini_api_key(self) -> Optional[str]:
        """Get the Google Gemini API key.

        Returns the embedding API key if provider is Gemini,
        otherwise falls back to the LLM API key if provider is Gemini,
        or returns None if neither is available.
        """
        if self.embedding.provider.lower() == "gemini" and self.embedding.api_key:
            return self.embedding.api_key
        elif self.llm.provider.lower() == "gemini" and self.llm.api_key:
            return self.llm.api_key
        return os.getenv("GEMINI_API_KEY")

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
        settings.embedding.api_key = os.getenv("EPIC_RAG_EMBEDDING_API_KEY")

        # Provider-specific model settings
        settings.embedding.openai_model = os.getenv(
            "EPIC_RAG_OPENAI_EMBEDDING_MODEL", settings.embedding.openai_model
        )
        settings.embedding.gemini_model = os.getenv(
            "EPIC_RAG_GEMINI_EMBEDDING_MODEL", settings.embedding.gemini_model
        )
        
        # Cache settings
        settings.embedding.cache.enabled = os.getenv(
            "EPIC_RAG_CACHE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        
        if memory_size := os.getenv("EPIC_RAG_CACHE_MEMORY_SIZE"):
            settings.embedding.cache.memory_size = int(memory_size)
            
        if expiration_days := os.getenv("EPIC_RAG_CACHE_EXPIRATION_DAYS"):
            settings.embedding.cache.expiration_days = int(expiration_days)
            
        settings.embedding.cache.clear_on_startup = os.getenv(
            "EPIC_RAG_CACHE_CLEAR_ON_STARTUP", "false"
        ).lower() in ("true", "1", "yes")

        # LLM settings
        settings.llm.provider = os.getenv(
            "EPIC_RAG_LLM_PROVIDER", settings.llm.provider
        )
        settings.llm.model = os.getenv("EPIC_RAG_LLM_MODEL", settings.llm.model)
        settings.llm.api_key = os.getenv("EPIC_RAG_LLM_API_KEY")

        return settings


# Default settings instance
settings = Settings.from_environment()
