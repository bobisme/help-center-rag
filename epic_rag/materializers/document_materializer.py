"""Materializer for Document objects."""

import json
import os
from typing import Dict, Type, Any, Optional, cast

import numpy as np
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from ..domain.models.document import Document


class DocumentMaterializer(BaseMaterializer):
    """Materializer for Document objects."""

    ASSOCIATED_TYPES = (Document,)
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
        self.chunks_path = os.path.join(self.uri, "chunks.json")
        self.viz_path = os.path.join(self.uri, "document_preview.txt")

    def load(self, data_type: Type[Document]) -> Document:
        """Load a Document from the artifact store.

        Args:
            data_type: The type of the data to load (Document)

        Returns:
            The loaded Document object
        """
        # Load document content
        content = ""
        if self.artifact_store.exists(self.content_path):
            with self.artifact_store.open(self.content_path, "r") as f:
                content = f.read()

        # Load metadata
        metadata_dict = {}
        if self.artifact_store.exists(self.metadata_path):
            with self.artifact_store.open(self.metadata_path, "r") as f:
                metadata_dict = json.load(f)

        # Extract the main document properties
        doc_id = metadata_dict.pop("id", None)
        title = metadata_dict.pop("title", "")
        epic_page_id = metadata_dict.pop("epic_page_id", None)
        epic_path = metadata_dict.pop("epic_path", None)
        created_at = metadata_dict.pop("created_at", None)
        updated_at = metadata_dict.pop("updated_at", None)

        # Load chunks if they exist
        chunks = []
        if self.artifact_store.exists(self.chunks_path):
            with self.artifact_store.open(self.chunks_path, "r") as f:
                chunks_data = json.load(f)
                # Note: We're not actually loading the chunks here
                # In a full implementation, we'd deserialize each chunk

        # Create the document
        document = Document(
            id=doc_id,
            title=title,
            content=content,
            epic_page_id=epic_page_id,
            epic_path=epic_path,
            metadata=metadata_dict,
        )

        # Set timestamps if available
        if created_at:
            document.created_at = created_at
        if updated_at:
            document.updated_at = updated_at

        return document

    def save(self, data: Document) -> None:
        """Save a Document to the artifact store.

        Args:
            data: The Document to save
        """
        # Save document content
        with self.artifact_store.open(self.content_path, "w") as f:
            f.write(data.content or "")

        # Prepare metadata
        metadata = {
            "id": data.id,
            "title": data.title,
            "epic_page_id": data.epic_page_id,
            "epic_path": data.epic_path,
        }

        # Handle timestamps which could be strings or datetime objects
        if hasattr(data, "created_at") and data.created_at:
            if hasattr(data.created_at, "isoformat"):
                metadata["created_at"] = data.created_at.isoformat()
            else:
                metadata["created_at"] = data.created_at

        if hasattr(data, "updated_at") and data.updated_at:
            if hasattr(data.updated_at, "isoformat"):
                metadata["updated_at"] = data.updated_at.isoformat()
            else:
                metadata["updated_at"] = data.updated_at

        # Add custom metadata
        if data.metadata:
            for key, value in data.metadata.items():
                metadata[key] = value

        # Save metadata
        with self.artifact_store.open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save chunks info (just IDs and indexes, not full chunks)
        if hasattr(data, "chunks") and data.chunks:
            chunks_data = [
                {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in data.chunks
            ]
            with self.artifact_store.open(self.chunks_path, "w") as f:
                json.dump(chunks_data, f, indent=2)

    def save_visualizations(self, data: Document) -> Dict[str, str]:
        """Save visualizations for the document.

        Args:
            data: The Document to visualize

        Returns:
            Dictionary of visualization paths
        """
        # Create a simple text visualization
        with self.artifact_store.open(self.viz_path, "w") as f:
            f.write(f"Document: {data.title}\n")
            f.write("-" * 80 + "\n")
            f.write(f"ID: {data.id}\n")
            f.write(f"Epic Page ID: {data.epic_page_id}\n")
            if data.metadata:
                f.write("\nMetadata:\n")
                for key, value in data.metadata.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\nContent Preview:\n")
            preview = (
                data.content[:500] + "..." if len(data.content) > 500 else data.content
            )
            f.write(preview)

        return {"document_preview": self.viz_path}

    def extract_metadata(self, data: Document) -> Dict[str, Any]:
        """Extract metadata from the document.

        Args:
            data: The Document to extract metadata from

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "title": data.title,
            "id": data.id,
            "epic_page_id": data.epic_page_id,
            "created_at": data.created_at.isoformat() if data.created_at else None,
            "updated_at": data.updated_at.isoformat() if data.updated_at else None,
        }

        # Add any custom metadata
        if data.metadata:
            metadata["custom_metadata"] = data.metadata

        # Add chunk information
        if hasattr(data, "chunks") and data.chunks:
            metadata["num_chunks"] = len(data.chunks)

        return metadata
