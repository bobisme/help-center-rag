#!/usr/bin/env python3
"""Test script for the Epic Documentation RAG system."""

import asyncio
from datetime import datetime
import torch

from epic_rag.domain.models.document import Document
from epic_rag.domain.models.retrieval import Query, ContextualRetrievalRequest
from epic_rag.infrastructure.config.settings import settings
from epic_rag.infrastructure.container import container, setup_container
from epic_rag.application.use_cases.ingest_document import IngestDocumentUseCase


async def test_embedding(provider="huggingface"):
    """Test the embedding service.

    Args:
        provider: The embedding provider to test ("huggingface", "openai", or "gemini")
    """
    # Backup original provider
    original_provider = settings.embedding.provider

    try:
        # Set the provider for this test
        settings.embedding.provider = provider

        # Re-setup container to use the specified provider
        setup_container()

        print(f"Testing {provider.capitalize()} embedding service...")
        try:
            embedding_service = container.get("embedding_service")
        except KeyError:
            print(f"Error: {provider.capitalize()} embedding service not available.")

            if provider.lower() == "huggingface":
                print("Make sure PyTorch and transformers are installed.")
            else:
                print(
                    f"Make sure {provider.upper()}_API_KEY is set in environment variables."
                )

            return None

        # Test embedding a simple text
        text = "Epic documentation is helpful for healthcare professionals."
        embedding = await embedding_service.embed_text(text)

        print(f"Text: {text}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        # Test similarity
        text2 = "Healthcare workers find Epic documentation useful."
        embedding2 = await embedding_service.embed_text(text2)

        similarity = await embedding_service.get_embedding_similarity(
            embedding, embedding2
        )
        print(f"Similarity between texts: {similarity:.4f}")

        return embedding

    finally:
        # Restore original provider
        settings.embedding.provider = original_provider
        setup_container()


async def test_document_ingestion():
    """Test document ingestion process."""
    print("\nTesting document ingestion...")

    # Get services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    chunking_service = container.get("chunking_service")
    embedding_service = container.get("embedding_service")

    # Create use case
    use_case = IngestDocumentUseCase(
        document_repository=document_repository,
        vector_repository=vector_repository,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )

    # Create test document
    doc_content = """# Epic Documentation Test

## Introduction

This is a test document for the Epic Documentation RAG system. The system is designed 
to process documentation for the Epic electronic health record system and make it 
searchable using semantic retrieval.

## Key Features

1. **Dynamic Chunking**: Documents are intelligently chunked based on content structure
2. **Vector Embedding**: Content is embedded using OpenAI models
3. **Contextual Retrieval**: Two-stage retrieval provides more accurate results
4. **Content Merging**: Related chunks are merged for better context

## Usage Examples

To ingest documents:

```
epic-rag ingest --source-dir /path/to/docs
```

To query the system:

```
epic-rag query "How do I create a new patient record?"
```

## Benefits

- Faster access to Epic documentation
- More accurate answers to questions
- Better context for complex queries
- Improved user experience

## Technical Architecture

The system follows Domain-Driven Design principles with separate layers for:

1. Domain models and interfaces
2. Application use cases
3. Infrastructure implementations
4. User interfaces
"""

    document = Document(
        title="Epic Documentation Test",
        content=doc_content,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "source": "test_script",
            "author": "Test User",
            "category": "Documentation",
        },
    )

    # Process the document
    result = await use_case.execute(
        document=document,
        dynamic_chunking=True,
        min_chunk_size=200,
        max_chunk_size=500,
    )

    print(f"Document processed with {len(result.chunks)} chunks")
    print("Chunk lengths:")
    for i, chunk in enumerate(result.chunks, 1):
        print(f"  Chunk {i}: {len(chunk.content)} chars")

    return result


async def test_retrieval():
    """Test retrieval functionality."""
    print("\nTesting retrieval...")

    # Get services
    retrieval_service = container.get("retrieval_service")
    embedding_service = container.get("embedding_service")

    # Create a test query
    query_text = "How do I use the Epic documentation system?"
    query = Query(text=query_text)

    # Embed the query
    query = await embedding_service.embed_query(query)

    print(f"Query: {query_text}")
    print(f"Embedding dimensions: {len(query.embedding) if query.embedding else 0}")

    # Create retrieval request
    request = ContextualRetrievalRequest(
        query=query,
        first_stage_k=10,
        second_stage_k=3,
        min_relevance_score=0.5,  # Lower threshold for testing
        use_query_context=True,
        merge_related_chunks=True,
    )

    # Execute retrieval
    result = await retrieval_service.contextual_retrieval(request)

    print(f"Retrieved {len(result.first_stage_results.chunks)} chunks in first stage")
    print(f"Final results: {len(result.final_chunks)} chunks")

    if result.merged_content:
        print("\nMerged Content Preview:")
        preview = (
            result.merged_content[:200] + "..."
            if len(result.merged_content) > 200
            else result.merged_content
        )
        print(preview)

    # Print performance metrics
    print(f"\nPerformance:")
    print(f"  Total latency: {result.total_latency_ms:.2f}ms")
    print(f"  Retrieval latency: {result.retrieval_latency_ms:.2f}ms")
    print(f"  Processing latency: {result.processing_latency_ms:.2f}ms")

    return result


async def main():
    """Run all tests."""
    print("Testing Epic Documentation RAG System\n")
    print(f"OpenAI API Key available: {'Yes' if settings.openai_api_key else 'No'}")
    print(f"Gemini API Key available: {'Yes' if settings.gemini_api_key else 'No'}")
    print(
        f"HuggingFace available: {'Yes (using CUDA)' if torch.cuda.is_available() else 'Yes (using CPU)'}"
    )

    # Setup the container
    setup_container()

    try:
        # Test HuggingFace embedding (local)
        await test_embedding("huggingface")

        # Test OpenAI embedding if available
        if settings.openai_api_key:
            await test_embedding("openai")

        # Test Gemini embedding if available
        if settings.gemini_api_key:
            await test_embedding("gemini")

        # Test document ingestion (uses whatever provider is set in settings)
        await test_document_ingestion()

        # Allow time for Qdrant to index
        print("\nWaiting for Qdrant to index documents...")
        await asyncio.sleep(2)

        # Test retrieval
        await test_retrieval()

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
