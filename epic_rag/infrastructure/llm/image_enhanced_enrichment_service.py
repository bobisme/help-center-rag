"""Implementation of contextual enrichment service with image descriptions."""

import asyncio
import os
import re
from typing import Dict, List, Set

from ...domain.models.document import Document, DocumentChunk
from ...domain.services.contextual_enrichment_service import ContextualEnrichmentService
from ...domain.services.image_description_service import ImageDescriptionService
from ...domain.services.llm_service import LLMService


class ImageEnhancedEnrichmentService(ContextualEnrichmentService):
    """Contextual enrichment service with image description capabilities."""

    def __init__(
        self,
        llm_service: LLMService,
        image_description_service: ImageDescriptionService,
        base_image_dir: str,
    ):
        """Initialize the service with LLM and image description services.

        Args:
            llm_service: The LLM service to use for context generation
            image_description_service: The service to use for generating image descriptions
            base_image_dir: Base directory where images are stored
        """
        self._llm_service = llm_service
        self._image_description_service = image_description_service
        self._base_image_dir = base_image_dir
        self._image_descriptions: Dict[str, str] = {}  # Cache of image descriptions

    async def enrich_chunk(
        self, document: Document, chunk: DocumentChunk
    ) -> DocumentChunk:
        """Enrich a document chunk with contextual information and image descriptions.

        Args:
            document: The parent document containing the chunk
            chunk: The document chunk to enrich

        Returns:
            The enriched document chunk
        """
        # First extract any image references in the chunk
        image_refs = self._extract_image_refs(chunk.content)

        # Try to match images with their descriptions
        image_descriptions = []
        if image_refs and self._image_descriptions:
            # The image references in the chunk content will be like "output/images/image.png"
            # But the keys in _image_descriptions may be different (like "path/to/output/images/image.png")
            # So we need to try different path combinations

            for image_ref in image_refs:
                # Try with full path
                if image_ref in self._image_descriptions:
                    desc = self._image_descriptions[image_ref]
                    if desc:
                        image_descriptions.append(f"Image: {desc}")
                        continue

                # Try with base filename
                img_base = os.path.basename(image_ref)
                for img_key, desc in self._image_descriptions.items():
                    if img_base in img_key and desc:
                        image_descriptions.append(f"Image: {desc}")
                        break

                # Try with full path addition
                full_path = os.path.join(self._base_image_dir, image_ref)
                if full_path in self._image_descriptions:
                    desc = self._image_descriptions[full_path]
                    if desc:
                        image_descriptions.append(f"Image: {desc}")

        # Create a prompt for the LLM for general context
        prompt = f"""<document> 
{document.content} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk.content} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Generate context using LLM
        context = await self._llm_service.generate_text(prompt, temperature=0.1)

        # Combine context and image descriptions if available
        final_context = context
        if image_descriptions:
            image_context = "\n".join(image_descriptions)
            final_context = f"{context}\n\n{image_context}"

        # Create a new chunk with the enhanced context prepended
        enriched_chunk = DocumentChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=f"{final_context}\n\n{chunk.content}",
            metadata={
                **chunk.metadata,
                "enriched": True,
                "context": context,
                "has_images": bool(image_refs),
                "image_count": len(image_refs),
                "image_descriptions": image_descriptions if image_descriptions else [],
                "image_refs": image_refs if image_refs else [],
            },
            embedding=chunk.embedding,
            chunk_index=chunk.chunk_index,
            previous_chunk_id=chunk.previous_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
            relevance_score=chunk.relevance_score,
        )

        return enriched_chunk

    async def enrich_chunks(
        self, document: Document, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Enrich multiple document chunks with contextual information and image descriptions.

        Args:
            document: The parent document containing the chunks
            chunks: The document chunks to enrich

        Returns:
            The enriched document chunks
        """
        # First, extract all unique image references from all chunks
        all_image_refs: Set[str] = set()
        for chunk in chunks:
            image_refs = self._extract_image_refs(chunk.content)
            all_image_refs.update(image_refs)

        # Process all images from the document to get descriptions
        if all_image_refs:
            try:
                # Fix paths in document content for more accurate extraction
                content = document.content.replace("![](output/images/", "![](")

                # Create a list of tuples with image paths and their surrounding context
                image_data = (
                    await self._image_description_service.extract_image_contexts(
                        content, self._base_image_dir
                    )
                )

                # Generate descriptions for all images
                self._image_descriptions = (
                    await self._image_description_service.generate_batch_descriptions(
                        image_data
                    )
                )
                print(f"Generated {len(self._image_descriptions)} image descriptions")

                # Print some details for debugging
                for img_path, desc in self._image_descriptions.items():
                    print(f"  - Image: {os.path.basename(img_path)}")
                    print(
                        f"    Description: {desc[:50]}..." if len(desc) > 50 else desc
                    )
            except Exception as e:
                print(f"Error generating image descriptions: {e}")
                self._image_descriptions = {}

        # Process chunks in parallel with image descriptions included
        enrichment_tasks = [self.enrich_chunk(document, chunk) for chunk in chunks]
        enriched_chunks = await asyncio.gather(*enrichment_tasks)

        return list(enriched_chunks)

    def _extract_image_refs(self, content: str) -> List[str]:
        """Extract image references from markdown content.

        Args:
            content: Markdown content

        Returns:
            List of image references
        """
        # Find all markdown image references
        image_pattern = r"!\[\]?\(([^)]+)\)"
        image_matches = re.finditer(image_pattern, content)
        return [match.group(1) for match in image_matches]
