"""Service registration using type-based dependency injection with Protocol classes."""

from typing import cast, Optional

from .container import container
from .config.settings import settings

# Import domain interfaces
from ..domain.repositories.document_repository import DocumentRepository
from ..domain.repositories.vector_repository import VectorRepository
from ..domain.services.chunking_service import ChunkingService
from ..domain.services.embedding_service import EmbeddingService
from ..domain.services.llm_service import LLMService
from ..domain.services.contextual_enrichment_service import ContextualEnrichmentService
from ..domain.services.retrieval_service import RetrievalService
from ..domain.services.reranker_service import RerankerService
from ..domain.services.image_description_service import ImageDescriptionService
from ..domain.services.lexical_search_service import LexicalSearchService
from ..domain.services.rank_fusion_service import RankFusionService

# Import implementations
from .persistence.sqlite_repository import SQLiteDocumentRepository
from .embedding.qdrant_repository import QdrantVectorRepository
from .document_processing.chunking_service import MarkdownChunkingService
from .embedding.openai_embedding_service import OpenAIEmbeddingService
from .embedding.gemini_embedding_service import GeminiEmbeddingService
from .embedding.huggingface_embedding_service import HuggingFaceEmbeddingService
from .embedding.cached_embedding_service import CachedEmbeddingService
from .llm.ollama_llm_service import OllamaLLMService
from .llm.contextual_enrichment_service import OllamaContextualEnrichmentService
from .llm.image_enhanced_enrichment_service import ImageEnhancedEnrichmentService
from .llm.ollama_image_description_service import OllamaImageDescriptionService
from .llm.smolvlm_image_description_service import SmolVLMImageDescriptionService
from .search.bm25s_search_service import BM25SSearchService
from .search.rank_fusion_service import RecipRankFusionService
from .reranker.cross_encoder_reranker_service import CrossEncoderRerankerService
from .retrieval.retrieval_service import ContextualRetrievalService
from ..application.use_cases.retrieve_context import RetrieveContextUseCase
from ..application.use_cases.ingest_document import IngestDocumentUseCase
from ..application.use_cases.answer_question import AnswerQuestionUseCase
from .embedding.embedding_cache import EmbeddingCache


# Type-based repository registrations
@container.register(DocumentRepository)
def create_document_repository() -> DocumentRepository:
    """Create the document repository implementation."""
    return SQLiteDocumentRepository(
        db_path=settings.database.path,
        enable_json=settings.database.enable_json,
    )


@container.register(VectorRepository)
def create_vector_repository() -> VectorRepository:
    """Create the vector repository implementation."""
    return QdrantVectorRepository(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
        distance=settings.qdrant.distance,
        local_path=settings.qdrant.local_path,
    )


# Service registrations
@container.register(ChunkingService)
def create_chunking_service() -> ChunkingService:
    """Create the chunking service implementation."""
    return MarkdownChunkingService()


@container.register(LLMService)
def create_llm_service() -> LLMService:
    """Create the LLM service implementation."""
    return OllamaLLMService(settings=settings.llm)


@container.register(ImageDescriptionService)
def create_image_description_service() -> ImageDescriptionService:
    """Create the image description service based on configuration."""
    if settings.llm.image_service_type == "smolvlm":
        return SmolVLMImageDescriptionService(
            settings=settings.llm,
            model_name=settings.llm.smolvlm_model,
            min_image_size=settings.llm.min_image_size,
        )
    else:
        return OllamaImageDescriptionService(
            settings=settings.llm,
            model=settings.llm.image_model,
            min_image_size=settings.llm.min_image_size,
        )


@container.register(ContextualEnrichmentService)
def create_contextual_enrichment_service(
    llm_service: LLMService,
    image_description_service: ImageDescriptionService,
) -> ContextualEnrichmentService:
    """Create the contextual enrichment service implementation.

    The dependencies are auto-injected by the container based on type.
    """
    if settings.llm.enable_image_enrichment:
        return ImageEnhancedEnrichmentService(
            llm_service=llm_service,
            image_description_service=image_description_service,
            base_image_dir=settings.images_dir,
        )
    else:
        return OllamaContextualEnrichmentService(llm_service=llm_service)


@container.register(EmbeddingCache)
def create_embedding_cache() -> EmbeddingCache:
    """Create the embedding cache."""
    return EmbeddingCache(
        settings=settings,
        memory_cache_size=settings.embedding.cache.memory_size,
        cache_expiration_days=settings.embedding.cache.expiration_days,
    )


@container.register(EmbeddingService)
def create_embedding_service(embedding_cache: EmbeddingCache) -> EmbeddingService:
    """Create the embedding service based on configuration."""
    # Create the base embedding service based on provider setting
    if settings.embedding.provider.lower() == "openai":
        base_service = OpenAIEmbeddingService(
            settings=settings,
            model=settings.embedding.openai_model,
            dimensions=settings.embedding.openai_dimensions,
            batch_size=settings.embedding.batch_size,
        )
    elif settings.embedding.provider.lower() == "gemini":
        base_service = GeminiEmbeddingService(
            settings=settings,
            model=settings.embedding.gemini_model,
            dimensions=settings.embedding.gemini_dimensions,
            batch_size=settings.embedding.batch_size,
        )
    elif settings.embedding.provider.lower() == "huggingface":
        base_service = HuggingFaceEmbeddingService(
            settings=settings,
            model_name=settings.embedding.huggingface_model,
            dimensions=settings.embedding.huggingface_dimensions,
            batch_size=settings.embedding.batch_size,
            device=settings.embedding.device,
            max_length=settings.embedding.max_length,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding.provider}"
        )

    # Wrap with caching if enabled
    if settings.embedding.cache.enabled:
        return CachedEmbeddingService(
            wrapped_service=base_service,
            settings=settings,
            provider_name=settings.embedding.provider,
            model_name=settings.embedding.model,
        )

    return base_service


@container.register(LexicalSearchService)
def create_lexical_search_service(
    document_repository: DocumentRepository,
) -> LexicalSearchService:
    """Create the lexical search service."""
    return BM25SSearchService(document_repository=document_repository)


@container.register(RankFusionService)
def create_rank_fusion_service() -> RankFusionService:
    """Create the rank fusion service."""
    return RecipRankFusionService(k=settings.retrieval.fusion_k)


# Only register reranker service if it's enabled in settings
if settings.retrieval.reranker.enabled:

    @container.register(RerankerService)
    def create_reranker_service() -> RerankerService:
        """Create the reranker service."""
        return CrossEncoderRerankerService(
            model_name=settings.retrieval.reranker.model_name
        )


@container.register(RetrievalService)
def create_retrieval_service(
    document_repository: DocumentRepository,
    vector_repository: VectorRepository,
    embedding_service: EmbeddingService,
    lexical_search_service: LexicalSearchService,
    rank_fusion_service: RankFusionService,
    llm_service: LLMService,
) -> RetrievalService:
    """Create the retrieval service."""
    # Get reranker service if registered, otherwise None
    reranker_service = None
    if settings.retrieval.reranker.enabled and container.has(RerankerService):
        reranker_service = container[RerankerService]

    return ContextualRetrievalService(
        document_repository=document_repository,
        vector_repository=vector_repository,
        embedding_service=embedding_service,
        bm25_service=lexical_search_service if settings.retrieval.enable_bm25 else None,
        rank_fusion_service=(
            rank_fusion_service if settings.retrieval.enable_bm25 else None
        ),
        llm_service=(
            llm_service if settings.retrieval.enable_query_transformation else None
        ),
        reranker_service=reranker_service,
        settings=settings,
    )


@container.register(IngestDocumentUseCase)
def create_ingest_document_use_case(
    document_repository: DocumentRepository,
    vector_repository: VectorRepository,
    chunking_service: ChunkingService,
    embedding_service: EmbeddingService,
    contextual_enrichment_service: ContextualEnrichmentService,
) -> IngestDocumentUseCase:
    """Create the ingest document use case."""
    return IngestDocumentUseCase(
        document_repository=document_repository,
        vector_repository=vector_repository,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        contextual_enrichment_service=contextual_enrichment_service,
    )


@container.register(RetrieveContextUseCase)
def create_retrieve_context_use_case(
    embedding_service: EmbeddingService,
    retrieval_service: RetrievalService,
) -> RetrieveContextUseCase:
    """Create the retrieve context use case."""
    return RetrieveContextUseCase(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
    )


@container.register(AnswerQuestionUseCase)
def create_answer_question_use_case(
    embedding_service: EmbeddingService,
    retrieval_service: RetrievalService,
    llm_service: LLMService,
) -> AnswerQuestionUseCase:
    """Create the answer question use case."""
    return AnswerQuestionUseCase(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )
