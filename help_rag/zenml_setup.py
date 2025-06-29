"""Setup for ZenML integration."""

from typing import cast, Type

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.materializers.materializer_registry import materializer_registry

from .materializers import (
    DocumentMaterializer,
    DocumentChunkMaterializer,
    EmbeddedChunkMaterializer,
)
from .domain.models.document import Document, DocumentChunk, EmbeddedChunk


def register_materializers():
    """Register all custom materializers with ZenML.

    This function attempts to register our custom materializers with ZenML's registry.
    If registration fails, we fall back to default serialization.
    """
    # Handle type checking by casting materializers to expected type
    document_materializer = cast(Type[BaseMaterializer], DocumentMaterializer)
    document_chunk_materializer = cast(
        Type[BaseMaterializer], DocumentChunkMaterializer
    )
    embedded_chunk_materializer = cast(
        Type[BaseMaterializer], EmbeddedChunkMaterializer
    )

    try:
        # Register materializers with correct parameter order (type, materializer)
        # This is the order expected by ZenML's type checking
        materializer_registry.register_materializer_type(
            Document, document_materializer
        )
        materializer_registry.register_materializer_type(
            DocumentChunk, document_chunk_materializer
        )
        materializer_registry.register_materializer_type(
            EmbeddedChunk, embedded_chunk_materializer
        )
        print("Registered custom materializers for Document types")
    except Exception as e:
        # If registration fails, log the error but continue
        # This aligns with the comment in components.py that materializer
        # registration isn't critical for pipeline function
        print(f"Materializer registration failed: {e}")
        print("Falling back to default pickle serialization for ZenML artifacts")
        return
