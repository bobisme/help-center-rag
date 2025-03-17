"""Dependency injection container."""

from typing import Dict, Any, Optional, Type

from .config.settings import settings


class ServiceContainer:
    """Container for managing service instances and dependencies."""

    def __init__(self):
        """Initialize an empty container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Any] = {}

    def register(self, name: str, instance: Any) -> None:
        """Register a service instance in the container.

        Args:
            name: The name of the service
            instance: The service instance
        """
        self._services[name] = instance

    def register_factory(self, name: str, factory_func) -> None:
        """Register a factory function for lazy instantiation.

        Args:
            name: The name of the service
            factory_func: A function that creates the service instance
        """
        self._factories[name] = factory_func

    def get(self, name: str) -> Any:
        """Get a service instance by name.

        If the service is not yet instantiated but has a factory,
        it will be created on first access.

        Args:
            name: The name of the service to retrieve

        Returns:
            The service instance

        Raises:
            KeyError: If the service is not registered
        """
        if name in self._services:
            return self._services[name]

        if name in self._factories:
            # Lazy instantiation
            instance = self._factories[name](self)
            self._services[name] = instance
            return instance

        raise KeyError(f"Service '{name}' not registered in container")

    def has(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: The name of the service

        Returns:
            True if the service is registered, False otherwise
        """
        return name in self._services or name in self._factories


# Create the global container instance
container = ServiceContainer()


def setup_container():
    """Set up the container with all service registrations."""
    from epic_rag.infrastructure.persistence.sqlite_repository import (
        SQLiteDocumentRepository,
    )
    from epic_rag.infrastructure.embedding.qdrant_repository import (
        QdrantVectorRepository,
    )
    from epic_rag.infrastructure.embedding.openai_embedding_service import (
        OpenAIEmbeddingService,
    )

    # Register repositories
    container.register_factory(
        "document_repository",
        lambda c: SQLiteDocumentRepository(
            db_path=settings.database.path,
            enable_json=settings.database.enable_json,
        ),
    )

    container.register_factory(
        "vector_repository",
        lambda c: QdrantVectorRepository(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            collection_name=settings.qdrant.collection_name,
            vector_size=settings.qdrant.vector_size,
            distance=settings.qdrant.distance,
            local_path=settings.qdrant.local_path,
        ),
    )

    # Register document processing services
    from epic_rag.infrastructure.document_processing.chunking_service import (
        MarkdownChunkingService,
    )

    container.register_factory("chunking_service", lambda c: MarkdownChunkingService())

    # Import embedding services
    from epic_rag.infrastructure.embedding.openai_embedding_service import (
        OpenAIEmbeddingService,
    )
    from epic_rag.infrastructure.embedding.gemini_embedding_service import (
        GeminiEmbeddingService,
    )
    from epic_rag.infrastructure.embedding.huggingface_embedding_service import (
        HuggingFaceEmbeddingService,
    )

    # Import caching wrapper
    from epic_rag.infrastructure.embedding.cached_embedding_service import (
        CachedEmbeddingService,
    )

    # Import LLM service
    from epic_rag.infrastructure.llm.ollama_llm_service import OllamaLLMService

    # Import BM25 search services
    from epic_rag.infrastructure.search.bm25_search_service import BM25SearchService
    from epic_rag.infrastructure.search.bm25s_search_service import BM25SSearchService

    # Import Rank Fusion service
    from epic_rag.infrastructure.search.rank_fusion_service import (
        RecipRankFusionService,
    )

    # Register LLM service
    container.register_factory(
        "llm_service",
        lambda c: OllamaLLMService(settings=settings.llm),
    )

    # Register BM25 search service based on configuration
    if settings.retrieval.bm25_implementation.lower() == "bm25s":
        container.register_factory(
            "bm25_search_service",
            lambda c: BM25SSearchService(
                document_repository=c.get("document_repository")
            ),
        )
    else:
        container.register_factory(
            "bm25_search_service",
            lambda c: BM25SearchService(
                document_repository=c.get("document_repository")
            ),
        )

    # Register Rank Fusion service
    container.register_factory(
        "rank_fusion_service",
        lambda c: RecipRankFusionService(k=settings.retrieval.fusion_k),
    )

    # Register embedding service based on provider configuration
    if settings.embedding.provider.lower() == "openai":
        container.register_factory(
            "base_embedding_service",
            lambda c: OpenAIEmbeddingService(
                settings=settings,
                model=settings.embedding.openai_model,
                dimensions=settings.embedding.openai_dimensions,
                batch_size=settings.embedding.batch_size,
            ),
        )

        # Wrap with caching if enabled
        if settings.embedding.cache.enabled:
            container.register_factory(
                "embedding_service",
                lambda c: CachedEmbeddingService(
                    wrapped_service=c.get("base_embedding_service"),
                    settings=settings,
                    provider_name="openai",
                    model_name=settings.embedding.openai_model,
                ),
            )
        else:
            container.register_factory(
                "embedding_service",
                lambda c: c.get("base_embedding_service"),
            )

    elif settings.embedding.provider.lower() == "gemini":
        container.register_factory(
            "base_embedding_service",
            lambda c: GeminiEmbeddingService(
                settings=settings,
                model=settings.embedding.gemini_model,
                dimensions=settings.embedding.gemini_dimensions,
                batch_size=settings.embedding.batch_size,
            ),
        )

        # Wrap with caching if enabled
        if settings.embedding.cache.enabled:
            container.register_factory(
                "embedding_service",
                lambda c: CachedEmbeddingService(
                    wrapped_service=c.get("base_embedding_service"),
                    settings=settings,
                    provider_name="gemini",
                    model_name=settings.embedding.gemini_model,
                ),
            )
        else:
            container.register_factory(
                "embedding_service",
                lambda c: c.get("base_embedding_service"),
            )

    elif settings.embedding.provider.lower() == "huggingface":
        container.register_factory(
            "base_embedding_service",
            lambda c: HuggingFaceEmbeddingService(
                settings=settings,
                model_name=settings.embedding.huggingface_model,
                dimensions=settings.embedding.huggingface_dimensions,
                batch_size=settings.embedding.batch_size,
                device=settings.embedding.device,
                max_length=settings.embedding.max_length,
            ),
        )

        # Wrap with caching if enabled
        if settings.embedding.cache.enabled:
            container.register_factory(
                "embedding_service",
                lambda c: CachedEmbeddingService(
                    wrapped_service=c.get("base_embedding_service"),
                    settings=settings,
                    provider_name="huggingface",
                    model_name=settings.embedding.huggingface_model,
                ),
            )
        else:
            container.register_factory(
                "embedding_service",
                lambda c: c.get("base_embedding_service"),
            )

    # Register retrieval service
    from epic_rag.infrastructure.retrieval.retrieval_service import (
        ContextualRetrievalService,
    )

    container.register_factory(
        "retrieval_service",
        lambda c: ContextualRetrievalService(
            document_repository=c.get("document_repository"),
            vector_repository=c.get("vector_repository"),
            embedding_service=c.get("embedding_service"),
            bm25_service=(
                c.get("bm25_search_service") if settings.retrieval.enable_bm25 else None
            ),
            rank_fusion_service=(
                c.get("rank_fusion_service") if settings.retrieval.enable_bm25 else None
            ),
            llm_service=(
                c.get("llm_service")
                if settings.retrieval.enable_query_transformation
                else None
            ),
            settings=settings,
        ),
    )
