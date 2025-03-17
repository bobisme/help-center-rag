"""Document chunking service."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models.document import Document, DocumentChunk


class ChunkingService(ABC):
    """Service for chunking documents into retrieval-optimized segments."""
    
    @abstractmethod
    async def chunk_document(
        self, 
        document: Document, 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Split a document into chunks based on the provided parameters.
        
        Args:
            document: The document to chunk
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            metadata: Additional metadata to add to each chunk
            
        Returns:
            List of document chunks with proper linking
        """
        pass
    
    @abstractmethod
    async def dynamic_chunk_document(
        self,
        document: Document,
        min_chunk_size: int = 300,
        max_chunk_size: int = 800,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Chunk a document with dynamic sizing based on content structure.
        
        This method attempts to create semantically meaningful chunks by
        respecting document structure such as headings, paragraphs, and lists.
        
        Args:
            document: The document to chunk
            min_chunk_size: Minimum size of each chunk in tokens
            max_chunk_size: Maximum size of each chunk in tokens
            metadata: Additional metadata to add to each chunk
            
        Returns:
            List of document chunks with proper linking
        """
        pass
    
    @abstractmethod
    async def merge_chunks(
        self,
        chunks: List[DocumentChunk],
        max_merged_size: int = 1500
    ) -> List[DocumentChunk]:
        """Merge related chunks to create more contextual units.
        
        Args:
            chunks: List of chunks to potentially merge
            max_merged_size: Maximum size of merged chunks
            
        Returns:
            List of merged chunks
        """
        pass