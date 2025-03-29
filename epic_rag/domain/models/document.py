"""Document domain models."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from .ident import new_id


@dataclass
class DocumentChunk:
    """A chunk of a document used for retrieval."""

    id: str = field(default_factory=lambda: new_id("chunk"))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # Reference to parent document
    document_id: str = ""
    chunk_index: int = 0

    # Contextual information
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # For contextual retrieval
    relevance_score: Optional[float] = None
    
    # Temporary attribute for vector database ID - won't be persisted in base class
    # Only used for type checking purposes - implementation in EmbeddedChunk
    # Declare this property to allow pyright to pass type checking
    vector_id: Optional[str] = None


@dataclass
class Document:
    """Represents a document in the system."""

    id: str = field(default_factory=lambda: new_id("doc"))
    title: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Epic-specific metadata
    epic_page_id: Optional[str] = None
    epic_path: Optional[str] = None

    # System metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    chunks: List[DocumentChunk] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in this document."""
        return len(self.chunks)


@dataclass
class EmbeddedChunk(DocumentChunk):
    """A document chunk with embedding data."""

    # Vector database metadata
    vector_id: Optional[str] = None
    indexed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        # Ensure embedding is not None
        if self.embedding is None:
            self.embedding = []
