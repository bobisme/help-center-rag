"""Materializer for DocumentChunk objects."""

import json
import os
from typing import Dict, Type, Any, Optional

from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from ..domain.models.document import DocumentChunk


class DocumentChunkMaterializer(BaseMaterializer):
    """Materializer for DocumentChunk objects."""

    ASSOCIATED_TYPES = (DocumentChunk,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def __init__(self, uri: str, artifact_store: Optional[Any] = None):
        """Initialize the materializer.

        Args:
            uri: The URI where the artifact data will be stored
            artifact_store: The artifact store used to store this artifact
        """
        super().__init__(uri, artifact_store)
        self.content_path = os.path.join(self.uri, "content.txt")
        self.metadata_path = os.path.join(self.uri, "metadata.json")
        self.viz_path = os.path.join(self.uri, "chunk_preview.txt")

    def load(self, data_type: Type[DocumentChunk]) -> DocumentChunk:
        """Load a DocumentChunk from the artifact store.

        Args:
            data_type: The type of the data to load (DocumentChunk)

        Returns:
            The loaded DocumentChunk object
        """
        # Load chunk content
        content = ""
        if self.artifact_store.exists(self.content_path):
            with self.artifact_store.open(self.content_path, "r") as f:
                content = f.read()

        # Load metadata
        metadata_dict = {}
        if self.artifact_store.exists(self.metadata_path):
            with self.artifact_store.open(self.metadata_path, "r") as f:
                metadata_dict = json.load(f)

        # Extract the main chunk properties
        chunk_id = metadata_dict.pop("id", None)
        document_id = metadata_dict.pop("document_id", None)
        chunk_index = metadata_dict.pop("chunk_index", 0)
        vector_id = metadata_dict.pop("vector_id", None)
        previous_chunk_id = metadata_dict.pop("previous_chunk_id", None)
        next_chunk_id = metadata_dict.pop("next_chunk_id", None)
        relevance_score = metadata_dict.pop("relevance_score", None)

        # Extract enrichment context if available
        context = metadata_dict.pop("context", None)

        # Create the chunk
        chunk = DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata_dict,
            chunk_index=chunk_index,
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            relevance_score=relevance_score,
        )

        # Store vector_id in metadata since it's not a field in DocumentChunk
        if vector_id is not None:
            chunk.metadata["vector_id"] = vector_id

        # Set context if available
        if context:
            chunk.context = context

        return chunk

    def save(self, data: DocumentChunk) -> None:
        """Save a DocumentChunk to the artifact store.

        Args:
            data: The DocumentChunk to save
        """
        # Save chunk content
        with self.artifact_store.open(self.content_path, "w") as f:
            f.write(data.content or "")

        # Prepare metadata
        metadata = {
            "id": data.id,
            "document_id": data.document_id,
            "chunk_index": data.chunk_index,
            "previous_chunk_id": getattr(data, "previous_chunk_id", None),
            "next_chunk_id": getattr(data, "next_chunk_id", None),
            "relevance_score": getattr(data, "relevance_score", None),
        }

        # Check if vector_id is in metadata
        if data.metadata and "vector_id" in data.metadata:
            metadata["vector_id"] = data.metadata["vector_id"]

        # Add contextual enrichment if present
        if hasattr(data, "context") and data.context:
            metadata["context"] = data.context

        # Add custom metadata
        if data.metadata:
            for key, value in data.metadata.items():
                metadata[key] = value

        # Save metadata
        with self.artifact_store.open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_visualizations(self, data: DocumentChunk) -> Dict[str, str]:
        """Save visualizations for the document chunk.

        Args:
            data: The DocumentChunk to visualize

        Returns:
            Dictionary of visualization paths
        """
        # Create a simple text visualization
        with self.artifact_store.open(self.viz_path, "w") as f:
            f.write(f"Document Chunk: {data.id}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Document ID: {data.document_id}\n")
            f.write(f"Chunk Index: {data.chunk_index}\n")
            if hasattr(data, "context") and data.context:
                f.write(f"\nContext: {data.context}\n")
            if data.metadata:
                f.write("\nMetadata:\n")
                for key, value in data.metadata.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\nContent Preview:\n")
            preview = (
                data.content[:500] + "..." if len(data.content) > 500 else data.content
            )
            f.write(preview)

        return {"chunk_preview": self.viz_path}

    def extract_metadata(self, data: DocumentChunk) -> Dict[str, Any]:
        """Extract metadata from the document chunk.

        Args:
            data: The DocumentChunk to extract metadata from

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "id": data.id,
            "document_id": data.document_id,
            "chunk_index": data.chunk_index,
            "has_context": hasattr(data, "context") and bool(data.context),
            "has_embedding": hasattr(data, "embedding") and data.embedding is not None,
            "content_length": len(data.content) if data.content else 0,
        }

        # Add vector ID if available
        if hasattr(data, "vector_id") and data.vector_id:
            metadata["vector_id"] = data.vector_id

        # Add relevance score if available
        if hasattr(data, "relevance_score") and data.relevance_score is not None:
            metadata["relevance_score"] = data.relevance_score

        return metadata
