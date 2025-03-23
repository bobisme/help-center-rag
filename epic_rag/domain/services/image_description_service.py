"""Interface for image description service."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...domain.models.document import DocumentChunk


class ImageDescriptionService(ABC):
    """Interface for a service that generates descriptions for images."""

    @abstractmethod
    async def generate_image_description(
        self, image_path: str, surrounding_text: str = ""
    ) -> str:
        """Generate a description for an image based on the image and surrounding text.

        Args:
            image_path: Path to the image file
            surrounding_text: Text surrounding the image in the document

        Returns:
            A description of the image
        """
        pass

    @abstractmethod
    async def generate_batch_descriptions(
        self, image_data: List[Tuple[str, str]]
    ) -> Dict[str, str]:
        """Generate descriptions for multiple images in batch.

        Args:
            image_data: List of tuples containing (image_path, surrounding_text)

        Returns:
            Dictionary mapping image paths to their descriptions
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def process_chunk_images(self, chunk: "DocumentChunk") -> "DocumentChunk":
        """Process a document chunk and add image descriptions within the content.

        This method looks for image references in the chunk content, generates
        descriptions for each image, and inserts those descriptions back into the content.

        Args:
            chunk: The document chunk to process

        Returns:
            Updated document chunk with image descriptions
        """
        pass
