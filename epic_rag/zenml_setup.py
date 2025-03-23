"""Setup for ZenML integration."""

from zenml.materializers.materializer_registry import materializer_registry

from .materializers import (
    DocumentMaterializer,
    DocumentChunkMaterializer,
    EmbeddedChunkMaterializer,
)
from .domain.models.document import Document, DocumentChunk, EmbeddedChunk


def register_materializers():
    """Register all custom materializers with ZenML."""
    materializer_registry.register_materializer_type(DocumentMaterializer, Document)
    materializer_registry.register_materializer_type(
        DocumentChunkMaterializer, DocumentChunk
    )
    materializer_registry.register_materializer_type(
        EmbeddedChunkMaterializer, EmbeddedChunk
    )

    print("Registered custom materializers for Document types")
