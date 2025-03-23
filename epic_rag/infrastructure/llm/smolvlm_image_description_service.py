"""Implementation of image description service using SmolVLM-Synthetic."""

import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from ...domain.services.image_description_service import ImageDescriptionService
from ..config.settings import LLMSettings


QUERY = """\
Describe this image from an insurance software documentation. Focus on UI
elements, functionality, and purpose. Context: %s"""


class SmolVLMImageDescriptionService(ImageDescriptionService):
    """Image description service that uses SmolVLM-Synthetic model."""

    def __init__(
        self,
        settings: LLMSettings,
        model_name: str = "HuggingFaceTB/SmolVLM-Synthetic",
        min_image_size: int = 64,
        device: Optional[str] = None,
    ):
        """Initialize the service.

        Args:
            settings: LLM settings
            model_name: The model name to use, defaults to
                HuggingFaceTB/SmolVLM-Synthetic
            min_image_size: Minimum size (width or height) in pixels for images
                to be processed
            device: Device to use (cuda, cpu, mps), if None will auto-detect
        """
        self._settings = settings
        self._model_name = model_name
        self._min_image_size = min_image_size

        # Auto-detect device if not provided
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        print(f"SmolVLM-Synthetic using device: {self._device}")

        # Initialize processor and model
        self._processor = AutoProcessor.from_pretrained(model_name)

        # First initialize the model, then move it to GPU
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
        )

        # Move model to device first, before setting attention implementation
        self._model = self._model.to(self._device)

        # Set Flash Attention only after moving to GPU
        if self._device == "cuda":
            self._model.config._attn_implementation = "flash_attention_2"
            print(f"Using Flash Attention 2.0 on {self._device}")

        # Set model to evaluation mode
        self._model.eval()

    async def generate_image_description(
        self, image_path: str, surrounding_text: str = ""
    ) -> str:
        """Generate a description for an image using SmolVLM-Synthetic.

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
                    # Load the image for processing
                    image = img.convert("RGB")
            except Exception as img_err:
                # If we can't determine size, continue anyway
                print(
                    f"Warning: Could not check image size for {image_path}: {img_err}"
                )
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    return f"[Unable to process image: {image_path}]"

            # Create prompt with surrounding text context
            query = QUERY.format(surrounding_text)

            # Create input messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": query}],
                },
            ]

            # Process image for inference - run in thread pool to avoid blocking
            def run_inference():
                # Prepare inputs
                prompt = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=prompt, images=[image], return_tensors="pt"
                )
                inputs = inputs.to(self._device)

                # Generate outputs
                with torch.no_grad():
                    generated_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=300,
                    )

                generated_text = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]

                # Extract the assistant's response
                assistant_start = generated_text.find("Assistant:")
                if assistant_start != -1:
                    start = assistant_start + len("Assistant:")
                    response = generated_text[start:].strip()
                else:
                    response = generated_text

                return response

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_inference)

            # Clean up any "The image shows" preamble if present
            result = result.replace("The image shows ", "")
            result = result.replace("The image displays ", "")

            return result

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
                # Skip empty results (which could happen if an image was
                # filtered out later)
                if result:
                    image_path = valid_image_data[i + j][0]
                    results[image_path] = result

        return results

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
            # image_title = match.group(1)
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
            # Format the description in a blockquote to ensure it's visible in
            # the console output
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
