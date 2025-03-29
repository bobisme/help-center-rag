"""Materializer for EmbeddedChunk objects."""

import json
import os
from typing import Dict, Type, Any, Optional

import numpy as np
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from ..domain.models.document import EmbeddedChunk


class EmbeddedChunkMaterializer(BaseMaterializer):
    """Materializer for EmbeddedChunk objects."""

    ASSOCIATED_TYPES = (EmbeddedChunk,)
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
        self.embedding_path = os.path.join(self.uri, "embedding.npy")
        self.viz_path = os.path.join(self.uri, "embedded_chunk_preview.txt")

    def load(self, data_type: Type[EmbeddedChunk]) -> EmbeddedChunk:
        """Load an EmbeddedChunk from the artifact store.

        Args:
            data_type: The type of the data to load (EmbeddedChunk)

        Returns:
            The loaded EmbeddedChunk object
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

        # Load embedding if available
        embedding = None
        if self.artifact_store.exists(self.embedding_path):
            with self.artifact_store.open(self.embedding_path, "rb") as f:
                embedding = np.load(f)

        # Create the embedded chunk
        chunk = EmbeddedChunk(
            id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata_dict,
            chunk_index=chunk_index,
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            relevance_score=relevance_score,
            embedding=embedding,
            vector_id=vector_id,
        )

        # Set context in metadata if available
        if context:
            chunk.metadata["context"] = context

        return chunk

    def save(self, data: EmbeddedChunk) -> None:
        """Save an EmbeddedChunk to the artifact store.

        Args:
            data: The EmbeddedChunk to save
        """
        # Save chunk content
        with self.artifact_store.open(self.content_path, "w") as f:
            f.write(data.content or "")

        # Prepare metadata
        metadata = {
            "id": data.id,
            "document_id": data.document_id,
            "chunk_index": data.chunk_index,
            "vector_id": getattr(data, "vector_id", None),
            "previous_chunk_id": getattr(data, "previous_chunk_id", None),
            "next_chunk_id": getattr(data, "next_chunk_id", None),
            "relevance_score": getattr(data, "relevance_score", None),
        }

        # Add contextual enrichment if present in metadata
        if data.metadata and "context" in data.metadata:
            metadata["context"] = data.metadata["context"]

        # Add custom metadata
        if data.metadata:
            for key, value in data.metadata.items():
                metadata[key] = value

        # Save metadata
        with self.artifact_store.open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save embedding if available
        if data.embedding is not None:
            with self.artifact_store.open(self.embedding_path, "wb") as f:
                np.save(f, data.embedding)

    def save_visualizations(self, data: EmbeddedChunk) -> Dict[str, Any]:
        """Save visualizations for the embedded chunk.

        Args:
            data: The EmbeddedChunk to visualize

        Returns:
            Dictionary of visualization paths
        """
        # Create a simple text visualization
        with self.artifact_store.open(self.viz_path, "w") as f:
            f.write(f"Embedded Chunk: {data.id}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Document ID: {data.document_id}\n")
            f.write(f"Chunk Index: {data.chunk_index}\n")

            if data.embedding is not None:
                f.write(f"Embedding Dimensions: {len(data.embedding)}\n")
                f.write(f"First 5 Values: {data.embedding[:5]}\n")
                f.write(f"Last 5 Values: {data.embedding[-5:]}\n")

            if data.metadata and "context" in data.metadata:
                f.write(f"\nContext: {data.metadata['context']}\n")

            if data.metadata:
                f.write("\nMetadata:\n")
                for key, value in data.metadata.items():
                    f.write(f"  {key}: {value}\n")

            f.write("\nContent Preview:\n")
            preview = (
                data.content[:500] + "..." if len(data.content) > 500 else data.content
            )
            f.write(preview)

        return {"embedded_chunk_preview": self.viz_path}

    def extract_metadata(self, data: EmbeddedChunk) -> Dict[str, Any]:
        """Extract metadata from the embedded chunk.

        Args:
            data: The EmbeddedChunk to extract metadata from

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "id": data.id,
            "document_id": data.document_id,
            "chunk_index": data.chunk_index,
            "has_context": data.metadata and "context" in data.metadata and bool(data.metadata["context"]),
            "content_length": len(data.content) if data.content else 0,
        }

        # Add embedding information if available
        if data.embedding is not None:
            metadata["embedding_dimensions"] = len(data.embedding)
            # Calculate some basic statistics about the embedding
            metadata["embedding_mean"] = float(np.mean(data.embedding))
            metadata["embedding_std"] = float(np.std(data.embedding))
            metadata["embedding_min"] = float(np.min(data.embedding))
            metadata["embedding_max"] = float(np.max(data.embedding))

        # Add vector ID if available
        if getattr(data, "vector_id", None) is not None:
            metadata["vector_id"] = data.vector_id

        # Add relevance score if available
        if hasattr(data, "relevance_score") and data.relevance_score is not None:
            metadata["relevance_score"] = data.relevance_score

        return metadata
