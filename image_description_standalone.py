#!/usr/bin/env python3
"""Standalone script for generating image descriptions and enriching chunks."""

import argparse
import asyncio
import os
import re
from typing import Dict, List, Tuple
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress

from epic_rag.domain.models.document import Document, DocumentChunk
from epic_rag.infrastructure.config.settings import settings
from epic_rag.infrastructure.llm.ollama_llm_service import OllamaLLMService
from epic_rag.infrastructure.llm.ollama_image_description_service import (
    OllamaImageDescriptionService,
)
from epic_rag.infrastructure.document_processing.chunking_service import (
    MarkdownChunkingService,
)


async def process_document_with_images(
    file_path: str, image_dir: str, min_image_size: int = 64
):
    """Process a document and its images.

    Args:
        file_path: Path to the markdown document
        image_dir: Directory containing images
        min_image_size: Minimum size for images to process

    Returns:
        Document text and dictionary of image descriptions
    """
    console = Console()

    # Read document content
    with open(file_path, "r") as f:
        content = f.read()

    # Extract image references
    image_pattern = r"!\[\]?\(([^)]+)\)"
    image_refs = re.findall(image_pattern, content)

    console.print(f"Found [bold]{len(image_refs)}[/bold] image references in document")

    # Create image description service
    image_service = OllamaImageDescriptionService(
        settings=settings.llm, model="gemma3:27b", min_image_size=min_image_size
    )

    # Fix content for better path extraction
    adjusted_content = content.replace("![](output/images/", "![](")

    # Extract image contexts
    image_data = await image_service.extract_image_contexts(adjusted_content, image_dir)

    console.print(f"Extracted [bold]{len(image_data)}[/bold] image contexts")

    # Generate descriptions
    with Progress() as progress:
        task = progress.add_task(
            "Generating image descriptions...", total=len(image_data)
        )

        # Process images in parallel
        descriptions = {}
        for path, context in image_data:
            # Generate description
            desc = await image_service.generate_image_description(path, context)
            if desc:
                descriptions[path] = desc
                console.print(
                    f"[green]✓[/green] {os.path.basename(path)}: {desc[:50]}..."
                    if len(desc) > 50
                    else desc
                )
            else:
                console.print(
                    f"[yellow]⚠[/yellow] {os.path.basename(path)}: No description generated (may be too small)"
                )

            progress.update(task, advance=1)

    return content, descriptions


async def main():
    """Run the standalone script."""
    parser = argparse.ArgumentParser(
        description="Standalone image description and chunking demo"
    )
    parser.add_argument(
        "document", type=str, help="Path to a markdown document with images"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=settings.images_dir,
        help="Directory containing images",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=64,
        help="Minimum size in pixels for images to process",
    )

    args = parser.parse_args()

    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]Image Description and Enrichment Demo[/bold blue]\n"
            f"Document: [green]{args.document}[/green]\n"
            f"Image directory: [cyan]{args.image_dir}[/cyan]",
            border_style="blue",
        )
    )

    try:
        # Process document and images
        content, image_descriptions = await process_document_with_images(
            args.document, args.image_dir, args.min_size
        )

        # Display summary of images processed
        console.print(
            f"\n[bold green]Generated descriptions for {len(image_descriptions)} images[/bold green]"
        )

        # Create a simulated chunk with images
        document = Document(
            id=str(uuid.uuid4()),
            title=os.path.basename(args.document),
            content=content,
            metadata={"source": args.document},
        )

        # Create chunking service
        chunking_service = MarkdownChunkingService()

        # Generate chunks
        chunks = await chunking_service.chunk_document(
            document, chunk_size=800, chunk_overlap=100
        )
        console.print(f"\n[bold]Generated {len(chunks)} chunks from document[/bold]")

        # Find chunks with images
        image_chunks = []
        for chunk in chunks:
            if "![" in chunk.content:
                image_chunks.append(chunk)

        if not image_chunks:
            console.print("[yellow]No chunks with images found[/yellow]")
            return

        console.print(
            f"[bold green]Found {len(image_chunks)} chunks containing images[/bold green]"
        )

        # Display a chunk with its images and image descriptions
        sample_chunk = image_chunks[0]
        console.print("\n[bold cyan]Sample Chunk Content:[/bold cyan]")
        console.print(Panel(sample_chunk.content, expand=False))

        # Extract images from this chunk
        chunk_image_refs = re.findall(r"!\[\]?\(([^)]+)\)", sample_chunk.content)

        # Display extracted image descriptions
        if chunk_image_refs:
            console.print("\n[bold cyan]Images in this chunk:[/bold cyan]")

            for img_ref in chunk_image_refs:
                # Try with different path combinations
                description = None

                # Direct match
                if img_ref in image_descriptions:
                    description = image_descriptions[img_ref]

                # Try with base filename
                if not description:
                    img_base = os.path.basename(img_ref)
                    for img_key, desc in image_descriptions.items():
                        if img_base in img_key:
                            description = desc
                            break

                # Try with full path
                if not description:
                    full_path = os.path.join(args.image_dir, img_ref)
                    if full_path in image_descriptions:
                        description = image_descriptions[full_path]

                if description:
                    console.print(f"[bold]Image:[/bold] {img_ref}")
                    console.print(Panel(description, border_style="green"))
                else:
                    console.print(f"[bold]Image:[/bold] {img_ref}")
                    console.print(
                        Panel("No description available", border_style="yellow")
                    )

        # Create a LLM service for context generation
        llm_service = OllamaLLMService(
            settings=settings.llm,
        )

        # Generate context for the sample chunk
        prompt = f"""<document> 
{document.content} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{sample_chunk.content} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        console.print("\n[bold cyan]Generating context for this chunk...[/bold cyan]")
        context = await llm_service.generate_text(prompt, temperature=0.1)

        # Combine context with image descriptions for this chunk
        chunk_image_descriptions = []
        for img_ref in chunk_image_refs:
            # Try with different path combinations
            description = None

            # Direct match
            if img_ref in image_descriptions:
                description = image_descriptions[img_ref]

            # Try with base filename
            if not description:
                img_base = os.path.basename(img_ref)
                for img_key, desc in image_descriptions.items():
                    if img_base in img_key:
                        description = desc
                        break

            # Try with full path
            if not description:
                full_path = os.path.join(args.image_dir, img_ref)
                if full_path in image_descriptions:
                    description = image_descriptions[full_path]

            if description:
                chunk_image_descriptions.append(f"Image: {description}")

        # Create the final enriched context with both text and image descriptions
        final_context = context
        if chunk_image_descriptions:
            image_context = "\n".join(chunk_image_descriptions)
            final_context = f"{context}\n\n{image_context}"

        # Show the final enriched chunk
        console.print("\n[bold cyan]Final Enriched Chunk:[/bold cyan]")
        enriched_chunk_content = f"{final_context}\n\n{sample_chunk.content}"

        # Split into sections for better display
        console.print(
            Panel(
                context,
                title="[bold yellow]General Context[/bold yellow]",
                border_style="yellow",
            )
        )

        if chunk_image_descriptions:
            console.print(
                Panel(
                    "\n".join(chunk_image_descriptions),
                    title=f"[bold green]Image Descriptions ({len(chunk_image_descriptions)})[/bold green]",
                    border_style="green",
                )
            )

        console.print(
            Panel(
                (
                    sample_chunk.content[:500] + "..."
                    if len(sample_chunk.content) > 500
                    else sample_chunk.content
                ),
                title="[bold blue]Original Chunk Content (truncated)[/bold blue]",
                border_style="blue",
                expand=False,
            )
        )

        # Explain how this would be incorporated into a chunk
        console.print("\n[bold]How image descriptions enhance chunks:[/bold]")
        console.print(
            """
1. Each image in a chunk gets a generated description using Gemma 27B
2. The descriptions are added to the chunk's context along with the general document context
3. This enriches the chunk with both textual context and understanding of images
4. When the chunk is embedded, the image information is included in the vector
5. Queries about visual elements or concepts shown in images can now match the chunks
        """
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
