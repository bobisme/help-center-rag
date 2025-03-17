"""ZenML custom components for the Epic Documentation RAG system."""

import os
from typing import Dict, Any, Optional, List

# Use the standard ZenML imports instead of trying to use RegistrableComponent
from zenml.stack import Stack
from zenml.artifact_stores import BaseArtifactStore

from ..config.settings import settings


class QdrantClient:
    """QdrantClient for ZenML.

    This is a wrapper around the Qdrant client to make it available in ZenML pipelines.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "epic_docs",
        distance_metric: str = "Cosine",
        vector_size: int = 1024,
    ):
        """Initialize the Qdrant client.

        Args:
            url: URL for the Qdrant server. If None, uses settings value.
            api_key: API key for the Qdrant server. If None, uses settings value.
            collection_name: Name of the collection to use.
            distance_metric: Distance metric to use for the collection.
            vector_size: Size of the vectors to store.
        """
        self.url = url or settings.qdrant.url or "http://localhost:6333"
        self.api_key = api_key or settings.qdrant.api_key
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.distance_metric = distance_metric
        self.vector_size = vector_size

    def get_client(self):
        """Return a Qdrant client instance."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            # Initialize the Qdrant client
            client = QdrantClient(url=self.url, api_key=self.api_key)

            # Ensure the collection exists
            self.ensure_collection_exists(client)

            return client
        except ImportError:
            raise ImportError(
                "Qdrant client is not installed. Please install it with: pip install qdrant-client"
            )

    def ensure_collection_exists(self, client):
        """Ensure the collection exists in Qdrant.

        Args:
            client: Qdrant client instance.
        """
        from qdrant_client.http.models import Distance, VectorParams

        # List all collections and check if ours exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Create the collection if it doesn't exist
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )


def register_custom_components():
    """Register all custom components with ZenML."""
    # In the newer ZenML versions, custom components can be registered
    # differently, so we're using a simpler approach by just creating
    # and returning a client
    return QdrantClient()
