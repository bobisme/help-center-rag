"""Qdrant implementation of the vector repository."""

# pyright: reportCallIssue=false
# pyright: reportArgumentType=false

import asyncio
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from ...domain.models.document import DocumentChunk, EmbeddedChunk
from ...domain.models.retrieval import Query
from ...domain.repositories.vector_repository import VectorRepository


class QdrantVectorRepository(VectorRepository):
    """Qdrant implementation of the vector repository."""

    def __init__(
        self,
        collection_name: str = "epic_docs",
        vector_size: int = 1536,
        distance: str = "Cosine",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_path: str = "qdrant_data",
    ):
        """Initialize the Qdrant repository.

        Args:
            collection_name: Name of the Qdrant collection
            vector_size: Size of the embedding vectors
            distance: Distance metric to use (Cosine, Dot, Euclid)
            url: URL of the Qdrant server (None for local)
            api_key: API key for Qdrant server
            local_path: Path for local Qdrant instance
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.url = url
        self.api_key = api_key
        self.local_path = local_path

        # Initialize client and collection
        self._initialize_client()
        asyncio.run(self._initialize_collection())

    def _initialize_client(self):
        """Initialize the Qdrant client."""
        if self.url:
            # Remote Qdrant instance
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
        else:
            # Local Qdrant instance
            self.client = QdrantClient(path=self.local_path, timeout=60)

    async def _initialize_collection(self):
        """Initialize the Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                # Create collection
                distance_config = {
                    "Cosine": qmodels.Distance.COSINE,
                    "Dot": qmodels.Distance.DOT,
                    "Euclid": qmodels.Distance.EUCLID,
                }

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=distance_config.get(
                            self.distance, qmodels.Distance.COSINE
                        ),
                    ),
                )

                # Create payload index for filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
        except Exception as e:
            print(f"Error initializing Qdrant collection: {str(e)}")
            raise

    async def store_embedding(self, chunk: EmbeddedChunk) -> str:
        """Store a chunk embedding in the vector database."""
        try:
            # Create the point
            point_id = chunk.id

            # Convert metadata and add document_id for filtering
            payload = dict(chunk.metadata)
            payload["document_id"] = chunk.document_id
            payload["chunk_index"] = chunk.chunk_index

            # Add the point to the collection
            # Ensure vector is not None
            vector = chunk.embedding if chunk.embedding is not None else []

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
                ],
            )

            return point_id
        except Exception as e:
            print(f"Error storing embedding: {str(e)}")
            raise

    async def delete_embedding(self, vector_id: str) -> bool:
        """Delete an embedding from the vector database."""
        try:
            # Delete the point
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=[vector_id]),
            )

            return True
        except Exception as e:
            print(f"Error deleting embedding: {str(e)}")
            return False

    async def search_similar(
        self, query: Query, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Search for similar chunks based on vector similarity."""
        try:
            # Create filter
            filter_obj = None
            if filters:
                filter_conditions = []

                if "document_id" in filters:
                    filter_conditions.append(
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=filters["document_id"]),
                        )
                    )

                if filter_conditions:
                    filter_obj = qmodels.Filter(must=filter_conditions)

            # Perform search - conditionally add filter only if we have filter conditions
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query.embedding,
                "limit": limit,
                "with_payload": True,
                "score_threshold": 0.0,
            }

            if filter_obj:
                search_params["filter"] = filter_obj

            results = self.client.search(**search_params)

            # Convert results to chunks
            chunks = []
            for result in results:
                # Create chunk with metadata from payload
                metadata = dict(result.payload)
                document_id = metadata.pop("document_id", "")
                chunk_index = metadata.pop("chunk_index", 0)

                chunk = DocumentChunk(
                    id=result.id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content="",  # Content needs to be fetched from document store
                    metadata=metadata,
                    relevance_score=result.score,
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            print(f"Error searching similar vectors: {str(e)}")
            return []

    async def batch_store_embeddings(self, chunks: List[EmbeddedChunk]) -> List[str]:
        """Store multiple embeddings at once."""
        try:
            # Create points
            points = []
            for chunk in chunks:
                # Convert metadata and add document_id for filtering
                payload = dict(chunk.metadata)
                payload["document_id"] = chunk.document_id
                payload["chunk_index"] = chunk.chunk_index

                # Ensure vector is not None
                vector = chunk.embedding if chunk.embedding is not None else []

                points.append(
                    qmodels.PointStruct(id=chunk.id, vector=vector, payload=payload)
                )

            # Add points to collection
            self.client.upsert(collection_name=self.collection_name, points=points)

            # Return vector IDs (same as chunk IDs)
            return [chunk.id for chunk in chunks]
        except Exception as e:
            print(f"Error batch storing embeddings: {str(e)}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            # Get collection info
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            # Extract stats - using more robust property access
            stats = {
                "vector_count": getattr(collection_info, "vectors_count", 0),
                "segment_count": len(getattr(collection_info, "segments_count", [])),
                "indexed_vector_count": getattr(
                    collection_info, "indexed_vectors_count", 0
                ),
                "size_bytes": 0,  # Can't get reliable disk size from API
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
            }

            return stats
        except UnexpectedResponse:
            # Collection might not exist yet
            return {
                "vector_count": 0,
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "error": "Collection not found",
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
