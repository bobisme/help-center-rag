"""Implementation of image description service using Ollama with Gemma 27B."""

import asyncio
import base64
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

import httpx
from ...domain.services.image_description_service import ImageDescriptionService
from ...domain.services.llm_service import LLMService
from ..config.settings import LLMSettings


class OllamaImageDescriptionService(ImageDescriptionService):
    """Image description service that uses Ollama with Gemma 27B."""

    def __init__(
        self, settings: LLMSettings, model: str = "gemma3:27b", min_image_size: int = 64
    ):
        """Initialize the service.

        Args:
            settings: LLM settings
            model: The specific model to use, defaulting to gemma3:27b
            min_image_size: Minimum size (width or height) in pixels for images to be processed
        """
        self._settings = settings
        self._model = model
        self._min_image_size = min_image_size
        self._api_base = "http://localhost:11434"
        self._client = httpx.AsyncClient(
            timeout=120.0
        )  # Longer timeout for image processing

    async def generate_image_description(
        self, image_path: str, surrounding_text: str = ""
    ) -> str:
        """Generate a description for an image using Gemma 27B.

        Args:
            image_path: Path to the image file
            surrounding_text: Text surrounding the image in the document

        Returns:
            A description of the image or empty string if image is too small
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return f"[Image not found: {image_path}]"

            # Check image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width < self._min_image_size and height < self._min_image_size:
                        # Skip small images like icons
                        return ""
            except Exception as img_err:
                # If we can't determine size, continue anyway
                print(
                    f"Warning: Could not check image size for {image_path}: {img_err}"
                )

            # Read and encode the image
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
                base64_image = base64.b64encode(image_data).decode("utf-8")

            # Create the prompt
            prompt = self._create_image_description_prompt(
                base64_image, surrounding_text
            )

            # Call Ollama API
            payload = {
                "model": self._model,
                "prompt": prompt,
                "temperature": 0.2,  # Slightly creative but mostly factual
                "stream": False,
                "options": {
                    "num_predict": 300,  # Limit token generation
                },
            }

            response = await self._client.post(
                f"{self._api_base}/api/generate", json=payload
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")

            result = response.json()
            return result.get("response", "").strip()

        except Exception as e:
            # Provide a fallback description in case of errors
            return f"[Image description unavailable. Error: {str(e)}]"

    async def generate_batch_descriptions(
        self, image_data: List[Tuple[str, str]]
    ) -> Dict[str, str]:
        """Generate descriptions for multiple images in batch.

        Args:
            image_data: List of tuples containing (image_path, surrounding_text)

        Returns:
            Dictionary mapping image paths to their descriptions
        """
        # Filter out any non-existent image paths
        valid_image_data = []
        for image_path, context in image_data:
            # Skip processing if file doesn't exist
            if not os.path.exists(image_path):
                continue

            # Check image dimensions at batch level to avoid redundant processing
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width >= self._min_image_size or height >= self._min_image_size:
                        valid_image_data.append((image_path, context))
                    # Silently skip small images
            except Exception as img_err:
                # If we can't determine size, include it just in case
                print(
                    f"Warning: Could not check image size for {image_path}: {img_err}"
                )
                valid_image_data.append((image_path, context))

        # Create tasks only for valid, sufficiently large images
        description_tasks = [
            self.generate_image_description(image_path, context)
            for image_path, context in valid_image_data
        ]

        # Process in batches to avoid overloading the system
        batch_size = 5
        results = {}

        for i in range(0, len(description_tasks), batch_size):
            batch = description_tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch)

            for j, result in enumerate(batch_results):
                # Skip empty results (which could happen if an image was filtered out later)
                if result:
                    image_path = valid_image_data[i + j][0]
                    results[image_path] = result

        return results

    async def extract_image_contexts(
        self, document_content: str, base_image_dir: str
    ) -> List[Tuple[str, str]]:
        """Extract image paths and their surrounding context from markdown content.

        Args:
            document_content: The markdown document content
            base_image_dir: Base directory where images are stored

        Returns:
            List of tuples containing (image_path, surrounding_text)
        """
        # Find all markdown image references
        image_pattern = r"!\[\]?\(([^)]+)\)"
        image_matches = re.finditer(image_pattern, document_content)

        image_contexts = []
        for match in image_matches:
            image_ref = match.group(1)
            # Fix path - don't add base_image_dir if the path already contains it
            if base_image_dir in image_ref:
                image_path = image_ref
            else:
                # Convert relative path to absolute
                image_path = os.path.join(base_image_dir, image_ref)

            # Get surrounding context - 200 chars before and after the image
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(document_content), match.end() + 200)
            context = document_content[start_pos:end_pos]

            image_contexts.append((image_path, context))

        return image_contexts

    async def process_chunk_images(self, chunk: "DocumentChunk") -> "DocumentChunk":
        """Process a document chunk and add image descriptions within the content.

        This method looks for image references in the chunk content, generates
        descriptions for each image, and inserts those descriptions back into the content.

        Args:
            chunk: The document chunk to process

        Returns:
            Updated document chunk with image descriptions
        """
        # Extract image references from content
        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        content = chunk.content
        base_image_dir = "output/images"

        # Extract all image matches
        matches = list(re.finditer(image_pattern, content))
        if not matches:
            return chunk  # No images to process

        # Track positions for inserting descriptions
        offset = 0
        processed_content = content

        # Process each image
        for match in matches:
            # Extract image path
            image_title = match.group(1)
            image_path = match.group(2)
            match_start = match.start() + offset
            match_end = match.end() + offset

            # Ensure the path is absolute
            if base_image_dir not in image_path:
                full_path = os.path.join(base_image_dir, image_path)
            else:
                full_path = image_path

            # Get context around the image
            context_start = max(0, match_start - 200)
            context_end = min(len(processed_content), match_end + 200)
            surrounding_text = processed_content[context_start:context_end]

            # Generate description
            description = await self.generate_image_description(
                full_path, surrounding_text
            )

            # Skip if no description (image too small or error)
            if not description:
                continue

            # Insert description after the image
            image_markdown = processed_content[match_start:match_end]
            # Format the description in a blockquote to ensure it's visible in the console output
            with_description = (
                f"{image_markdown}\n\n> **Image description:** {description}\n"
            )

            # Replace the original image with the image + description
            processed_content = (
                processed_content[:match_start]
                + with_description
                + processed_content[match_end:]
            )

            # Update offset for future matches
            offset += len(with_description) - len(image_markdown)

        # Create a new chunk with the enhanced content
        # Preserve all original properties
        return chunk.__class__(
            id=chunk.id,
            document_id=chunk.document_id,
            content=processed_content,
            metadata=chunk.metadata,
            embedding=chunk.embedding,
            chunk_index=chunk.chunk_index,
            previous_chunk_id=getattr(chunk, "previous_chunk_id", None),
            next_chunk_id=getattr(chunk, "next_chunk_id", None),
            relevance_score=getattr(chunk, "relevance_score", None),
        )

    def _create_image_description_prompt(
        self, base64_image: str, surrounding_text: str
    ) -> str:
        """Create a prompt for the image description task.

        Args:
            base64_image: Base64 encoded image
            surrounding_text: Text surrounding the image

        Returns:
            The prompt for the image description task
        """
        return f"""I'll show you an image from Applied Epic insurance agency management software documentation. 
Please provide a concise description of what you see in the image.

Context about the image from surrounding text in the document:
{surrounding_text}

Image: <base64>{base64_image}</base64>

Describe the image in 1-3 sentences. Focus on:
1. What UI element or feature is shown
2. The purpose or function of what's displayed
3. Any relevant buttons, fields, or interactive elements visible

Provide ONLY the description - no preamble, analysis, or additional commentary.
"""
