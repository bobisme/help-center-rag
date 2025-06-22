"""Retrieval domain models."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

from .document import DocumentChunk


@dataclass
class Query:
    """Represents a user query to the system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    transformed_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query_id: str = ""
    chunks: List[DocumentChunk] = field(default_factory=list)
    merged_content: Optional[str] = None

    # Metrics
    latency_ms: Optional[float] = None

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a chunk to the retrieval result."""
        self.chunks.append(chunk)


@dataclass
class ContextualRetrievalRequest:
    """Request for contextual retrieval."""

    query: Query
    first_stage_k: int = 20  # Number of documents to retrieve in first stage
    second_stage_k: int = 5  # Number of documents to retrieve in second stage
    min_relevance_score: float = 0.7

    # Filtering
    filter_metadata: Dict[str, Any] = field(default_factory=dict)

    # Context awareness
    use_query_context: bool = True
    merge_related_chunks: bool = True


@dataclass
class ContextualRetrievalResult:
    """Result of a contextual retrieval operation."""

    query: Query
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Multi-stage retrieval results
    first_stage_results: RetrievalResult = field(default_factory=RetrievalResult)
    second_stage_results: Optional[RetrievalResult] = None

    # Final filtered and processed result
    final_chunks: List[DocumentChunk] = field(default_factory=list)
    merged_content: Optional[str] = None

    # Performance metrics
    total_latency_ms: Optional[float] = None
    retrieval_latency_ms: Optional[float] = None
    processing_latency_ms: Optional[float] = None
