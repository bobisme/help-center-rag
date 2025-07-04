"""Chunking service implementation."""

import re
from typing import List, Dict, Any, Optional
import uuid

from ...domain.models.document import Document, DocumentChunk
from ...domain.services.chunking_service import ChunkingService


class MarkdownChunkingService(ChunkingService):
    """Chunking service implementation for Markdown documents."""

    async def chunk_document(
        self,
        document: Document,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split a document into fixed-size chunks with overlap."""
        # Get document content
        content = document.content

        # Split by paragraphs first for more natural chunking
        paragraphs = [p for p in re.split(r"\n\s*\n", content) if p.strip()]

        # Process paragraphs into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            # Count tokens in paragraph (simple word count)
            para_size = len(re.findall(r"\S+", para))

            # If adding this paragraph would exceed chunk_size, finalize current chunk
            if current_size + para_size > chunk_size and current_size > 0:
                # Create chunk from current paragraphs
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    self._create_chunk(
                        document,
                        chunk_content,
                        len(chunks),
                        {
                            **(metadata or {}),
                            "chunk_method": "fixed",
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "tokens": current_size,
                        },
                    )
                )

                # Handle overlap: find paragraphs to include in next chunk for overlap
                overlap_tokens = 0
                overlap_paras = []

                # Go backwards through paragraphs to find overlap
                for para in reversed(current_chunk):
                    para_tokens = len(re.findall(r"\S+", para))
                    if overlap_tokens + para_tokens <= chunk_overlap:
                        overlap_paras.insert(0, para)
                        overlap_tokens += para_tokens
                    else:
                        break

                # Reset for next chunk, but include overlap paragraphs
                current_chunk = overlap_paras.copy()
                current_size = overlap_tokens

            # Add paragraph to current chunk
            current_chunk.append(para)
            current_size += para_size

        # Add the final chunk if anything remains
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(
                self._create_chunk(
                    document,
                    chunk_content,
                    len(chunks),
                    {
                        **(metadata or {}),
                        "chunk_method": "fixed",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "tokens": current_size,
                    },
                )
            )

        # Link chunks together
        for i in range(1, len(chunks)):
            chunks[i].previous_chunk_id = chunks[i - 1].id
            chunks[i - 1].next_chunk_id = chunks[i].id

        return chunks

    async def dynamic_chunk_document(
        self,
        document: Document,
        min_chunk_size: int = 300,
        max_chunk_size: int = 800,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Chunk a document with dynamic sizing based on content structure."""
        # Get document content
        content = document.content

        # Split by headings
        heading_pattern = r"(?:^|\n)(#{1,6})\s+(.+)(?:\n|$)"

        # Collect all heading positions
        headings = [
            (m.start(), m.group(1), m.group(2))
            for m in re.finditer(heading_pattern, content)
        ]

        if not headings:
            # If no headings, we need a different approach
            # Split by paragraphs and chunk appropriately
            paragraphs = [p for p in re.split(r"\n\s*\n", content) if p.strip()]

            # If there are very few paragraphs, use fixed chunking
            if len(paragraphs) <= 3:
                return await self.chunk_document(
                    document=document,
                    chunk_size=max_chunk_size,
                    chunk_overlap=min(100, max_chunk_size // 10),
                    metadata=metadata,
                )

            # Otherwise, build chunks from paragraphs
            chunks = []
            current_chunk = []
            current_size = 0

            for para in paragraphs:
                para_size = len(re.findall(r"\S+", para))

                # If adding this paragraph would exceed max_chunk_size, finalize current chunk
                if (
                    current_size + para_size > max_chunk_size
                    and current_size >= min_chunk_size
                ):
                    # Create chunk from current paragraphs
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(
                        self._create_chunk(
                            document,
                            chunk_content,
                            len(chunks),
                            {
                                **(metadata or {}),
                                "chunk_method": "dynamic_paragraphs",
                                "min_chunk_size": min_chunk_size,
                                "max_chunk_size": max_chunk_size,
                                "tokens": current_size,
                                "titles": [document.title] if document.title else [],
                            },
                        )
                    )

                    # Reset for next chunk
                    current_chunk = []
                    current_size = 0

                # Handle very large paragraphs that exceed max_chunk_size on their own
                if para_size > max_chunk_size:
                    # If we have accumulated content, save it first
                    if current_chunk:
                        chunk_content = "\n\n".join(current_chunk)
                        chunks.append(
                            self._create_chunk(
                                document,
                                chunk_content,
                                len(chunks),
                                {
                                    **(metadata or {}),
                                    "chunk_method": "dynamic_paragraphs",
                                    "min_chunk_size": min_chunk_size,
                                    "max_chunk_size": max_chunk_size,
                                    "tokens": current_size,
                                    "titles": (
                                        [document.title] if document.title else []
                                    ),
                                },
                            )
                        )

                        # Reset for next chunk
                        current_chunk = []
                        current_size = 0

                    # Split this large paragraph into sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sent_chunk = []
                    sent_size = 0

                    for sent in sentences:
                        sent_tokens = len(re.findall(r"\S+", sent))

                        if (
                            sent_size + sent_tokens > max_chunk_size
                            and sent_size >= min_chunk_size
                        ):
                            # Create a chunk from the accumulated sentences
                            sent_content = " ".join(sent_chunk)
                            chunks.append(
                                self._create_chunk(
                                    document,
                                    sent_content,
                                    len(chunks),
                                    {
                                        **(metadata or {}),
                                        "chunk_method": "dynamic_sentences",
                                        "min_chunk_size": min_chunk_size,
                                        "max_chunk_size": max_chunk_size,
                                        "tokens": sent_size,
                                        "titles": (
                                            [document.title] if document.title else []
                                        ),
                                    },
                                )
                            )

                            # Reset for next sentence chunk
                            sent_chunk = []
                            sent_size = 0

                        # Add sentence to current chunk
                        sent_chunk.append(sent)
                        sent_size += sent_tokens

                    # Add any remaining sentences as a chunk
                    if sent_chunk:
                        sent_content = " ".join(sent_chunk)
                        chunks.append(
                            self._create_chunk(
                                document,
                                sent_content,
                                len(chunks),
                                {
                                    **(metadata or {}),
                                    "chunk_method": "dynamic_sentences",
                                    "min_chunk_size": min_chunk_size,
                                    "max_chunk_size": max_chunk_size,
                                    "tokens": sent_size,
                                    "titles": (
                                        [document.title] if document.title else []
                                    ),
                                },
                            )
                        )

                    # Skip adding this paragraph to current_chunk since we've already processed it
                    continue

                # Add paragraph to current chunk
                current_chunk.append(para)
                current_size += para_size

            # Add the final chunk if anything remains
            if current_chunk:
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    self._create_chunk(
                        document,
                        chunk_content,
                        len(chunks),
                        {
                            **(metadata or {}),
                            "chunk_method": "dynamic_paragraphs",
                            "min_chunk_size": min_chunk_size,
                            "max_chunk_size": max_chunk_size,
                            "tokens": current_size,
                            "titles": [document.title] if document.title else [],
                        },
                    )
                )

            # Link chunks together
            for i in range(1, len(chunks)):
                chunks[i].previous_chunk_id = chunks[i - 1].id
                chunks[i - 1].next_chunk_id = chunks[i].id

            return chunks

        # Create sections based on headings
        sections = []
        for i, (start_pos, level, title) in enumerate(headings):
            # Determine end position
            if i < len(headings) - 1:
                end_pos = headings[i + 1][0]
            else:
                end_pos = len(content)

            # Extract section content
            section_content = content[start_pos:end_pos]

            # Add to sections
            sections.append((section_content, level, title))

        # Handle content before first heading
        if headings[0][0] > 0:
            first_content = content[: headings[0][0]].strip()
            if first_content:
                sections.insert(0, (first_content, "", "Introduction"))

        # Group sections into chunks based on size and heading level
        chunks = []
        current_chunk = []
        current_size = 0
        current_titles = []

        for section_content, level, title in sections:
            # Calculate section size
            section_size = len(re.findall(r"\S+", section_content))

            # If this is a top-level heading (# or ##) and we already have content,
            # finalize the current chunk unless it's too small
            if (
                level in ["#", "##"]
                and current_size > 0
                and current_size >= min_chunk_size
            ):
                # Create chunk from current sections
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    self._create_chunk(
                        document,
                        chunk_content,
                        len(chunks),
                        {
                            **(metadata or {}),
                            "chunk_method": "dynamic",
                            "min_chunk_size": min_chunk_size,
                            "max_chunk_size": max_chunk_size,
                            "tokens": current_size,
                            "titles": current_titles,
                        },
                    )
                )

                # Reset for next chunk
                current_chunk = []
                current_size = 0
                current_titles = []

            # If adding this section would exceed max_chunk_size and we're not at min_chunk_size yet,
            # split the section further into paragraphs
            if current_size + section_size > max_chunk_size:
                # Only finalize the chunk if it has content and meets minimum size
                if current_chunk and current_size >= min_chunk_size:
                    # Create chunk from current sections
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(
                        self._create_chunk(
                            document,
                            chunk_content,
                            len(chunks),
                            {
                                **(metadata or {}),
                                "chunk_method": "dynamic",
                                "min_chunk_size": min_chunk_size,
                                "max_chunk_size": max_chunk_size,
                                "tokens": current_size,
                                "titles": current_titles,
                            },
                        )
                    )

                    # Reset for next chunk
                    current_chunk = []
                    current_size = 0
                    current_titles = []
                # If section itself exceeds max_chunk_size, we need to split it into paragraphs
                elif section_size > max_chunk_size:
                    # Split this section into paragraphs
                    section_paras = [
                        p for p in re.split(r"\n\s*\n", section_content) if p.strip()
                    ]

                    # Process each paragraph
                    para_chunk = []
                    para_size = 0
                    para_titles = [title] if title else []

                    for para in section_paras:
                        para_tokens = len(re.findall(r"\S+", para))

                        if (
                            para_size + para_tokens > max_chunk_size
                            and para_size >= min_chunk_size
                        ):
                            # Create a chunk from the accumulated paragraphs
                            para_content = "\n\n".join(para_chunk)
                            chunks.append(
                                self._create_chunk(
                                    document,
                                    para_content,
                                    len(chunks),
                                    {
                                        **(metadata or {}),
                                        "chunk_method": "dynamic_paragraph",
                                        "min_chunk_size": min_chunk_size,
                                        "max_chunk_size": max_chunk_size,
                                        "tokens": para_size,
                                        "titles": para_titles,
                                    },
                                )
                            )

                            # Reset for next paragraph chunk
                            para_chunk = []
                            para_size = 0

                        # Add paragraph to current chunk
                        para_chunk.append(para)
                        para_size += para_tokens

                    # Add any remaining paragraphs as a chunk
                    if para_chunk and (para_size >= min_chunk_size or not chunks):
                        para_content = "\n\n".join(para_chunk)
                        chunks.append(
                            self._create_chunk(
                                document,
                                para_content,
                                len(chunks),
                                {
                                    **(metadata or {}),
                                    "chunk_method": "dynamic_paragraph",
                                    "min_chunk_size": min_chunk_size,
                                    "max_chunk_size": max_chunk_size,
                                    "tokens": para_size,
                                    "titles": para_titles,
                                },
                            )
                        )

                    # Continue to next section without adding the current one
                    continue

            # Add section to current chunk
            current_chunk.append(section_content)
            current_size += section_size
            if title and title not in current_titles:
                current_titles.append(title)

        # Add the final chunk if anything remains
        if current_chunk:
            # Check if the final chunk exceeds max_chunk_size
            if current_size > max_chunk_size:
                # Need to split the final chunk into paragraphs
                all_content = "\n\n".join(current_chunk)
                final_paras = [
                    p for p in re.split(r"\n\s*\n", all_content) if p.strip()
                ]

                # Process paragraphs into smaller chunks
                para_chunk = []
                para_size = 0

                for para in final_paras:
                    para_tokens = len(re.findall(r"\S+", para))

                    if (
                        para_size + para_tokens > max_chunk_size
                        and para_size >= min_chunk_size
                    ):
                        # Create a chunk from accumulated paragraphs
                        para_content = "\n\n".join(para_chunk)
                        chunks.append(
                            self._create_chunk(
                                document,
                                para_content,
                                len(chunks),
                                {
                                    **(metadata or {}),
                                    "chunk_method": "dynamic_paragraph_final",
                                    "min_chunk_size": min_chunk_size,
                                    "max_chunk_size": max_chunk_size,
                                    "tokens": para_size,
                                    "titles": current_titles,
                                },
                            )
                        )

                        # Reset for next paragraph chunk
                        para_chunk = []
                        para_size = 0

                    # Add paragraph to current chunk
                    para_chunk.append(para)
                    para_size += para_tokens

                # Add any remaining paragraphs as a chunk
                if para_chunk:
                    para_content = "\n\n".join(para_chunk)
                    chunks.append(
                        self._create_chunk(
                            document,
                            para_content,
                            len(chunks),
                            {
                                **(metadata or {}),
                                "chunk_method": "dynamic_paragraph_final",
                                "min_chunk_size": min_chunk_size,
                                "max_chunk_size": max_chunk_size,
                                "tokens": para_size,
                                "titles": current_titles,
                            },
                        )
                    )
            else:
                # Add as a single chunk if it's within size limits
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    self._create_chunk(
                        document,
                        chunk_content,
                        len(chunks),
                        {
                            **(metadata or {}),
                            "chunk_method": "dynamic",
                            "min_chunk_size": min_chunk_size,
                            "max_chunk_size": max_chunk_size,
                            "tokens": current_size,
                            "titles": current_titles,
                        },
                    )
                )

        # Link chunks together
        for i in range(1, len(chunks)):
            chunks[i].previous_chunk_id = chunks[i - 1].id
            chunks[i - 1].next_chunk_id = chunks[i].id

        return chunks

    async def merge_chunks(
        self, chunks: List[DocumentChunk], max_merged_size: int = 1500
    ) -> List[DocumentChunk]:
        """Merge related chunks to create more contextual units."""
        if not chunks:
            return []

        # Group chunks by document_id
        chunks_by_doc = {}
        for chunk in chunks:
            if chunk.document_id not in chunks_by_doc:
                chunks_by_doc[chunk.document_id] = []
            chunks_by_doc[chunk.document_id].append(chunk)

        # Merge chunks within each document
        merged_chunks = []

        for doc_id, doc_chunks in chunks_by_doc.items():
            # Sort chunks by index
            doc_chunks.sort(key=lambda c: c.chunk_index)

            # Group chunks for merging
            current_chunks = []
            current_size = 0

            for chunk in doc_chunks:
                # Get chunk size
                chunk_size = chunk.metadata.get(
                    "tokens", len(re.findall(r"\S+", chunk.content))
                )

                # If adding this chunk would exceed max_merged_size, finalize current merged chunk
                if current_size + chunk_size > max_merged_size and current_chunks:
                    # Create merged chunk
                    merged_chunk = DocumentChunk(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        content="\n\n".join(c.content for c in current_chunks),
                        metadata={
                            "chunk_method": "merged",
                            "original_chunks": len(current_chunks),
                            "merged_size": current_size,
                            "document_title": current_chunks[0].metadata.get(
                                "document_title", ""
                            ),
                        },
                        chunk_index=current_chunks[0].chunk_index,
                    )

                    # Add title if available in first chunk
                    if "title" in current_chunks[0].metadata:
                        merged_chunk.metadata["title"] = current_chunks[0].metadata[
                            "title"
                        ]

                    merged_chunks.append(merged_chunk)

                    # Reset for next merge group
                    current_chunks = []
                    current_size = 0

                # Add chunk to current group
                current_chunks.append(chunk)
                current_size += chunk_size

            # Add final merged chunk if any chunks remain
            if current_chunks:
                merged_chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    content="\n\n".join(c.content for c in current_chunks),
                    metadata={
                        "chunk_method": "merged",
                        "original_chunks": len(current_chunks),
                        "merged_size": current_size,
                        "document_title": current_chunks[0].metadata.get(
                            "document_title", ""
                        ),
                    },
                    chunk_index=current_chunks[0].chunk_index,
                )

                # Add title if available in first chunk
                if "title" in current_chunks[0].metadata:
                    merged_chunk.metadata["title"] = current_chunks[0].metadata["title"]

                merged_chunks.append(merged_chunk)

        return merged_chunks

    def _create_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        metadata: Dict[str, Any],
    ) -> DocumentChunk:
        """Create a new chunk with the given content and metadata."""
        # Add document title to metadata
        if "document_title" not in metadata:
            metadata["document_title"] = document.title

        # Try to extract a title from the first line
        lines = content.split("\n", 1)
        if lines and lines[0].startswith("# "):
            title = lines[0].replace("# ", "")
            metadata["title"] = title
        elif "title" not in metadata:
            metadata["title"] = document.title

        return DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document.id,
            content=content,
            metadata=metadata,
            chunk_index=chunk_index,
        )
