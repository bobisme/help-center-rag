#!/usr/bin/env python3
"""Script to show full enriched chunks for a document file."""

import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from epic_rag.domain.models.document import Document, DocumentChunk
from epic_rag.infrastructure.config.settings import settings
from epic_rag.infrastructure.llm.ollama_llm_service import OllamaLLMService
from epic_rag.infrastructure.llm.ollama_image_description_service import (
    OllamaImageDescriptionService,
)
from epic_rag.infrastructure.llm.image_enhanced_enrichment_service import (
    ImageEnhancedEnrichmentService,
)
from epic_rag.infrastructure.document_processing.chunking_service import (
    MarkdownChunkingService,
)


async def process_document(
    file_path: str,
    base_image_dir: str,
    llm_model: str,
    image_model: str,
    min_image_size: int,
    chunk_size: int,
    chunk_overlap: int,
    enhance_images: bool,
) -> Tuple[Document, List[DocumentChunk]]:
    """Process document and generate enriched chunks.

    Args:
        file_path: Path to document file
        base_image_dir: Path to image directory
        llm_model: Model to use for text enrichment
        image_model: Model to use for image description
        min_image_size: Minimum size of images to process
        chunk_size: Size of chunks to generate
        chunk_overlap: Overlap between chunks
        enhance_images: Whether to use image enhancement

    Returns:
        Tuple of document and enriched chunks
    """
    console = Console()

    # Read document content
    with open(file_path, "r") as f:
        content = f.read()

    # Create document
    document_id = str(uuid.uuid4())
    document = Document(
        id=document_id,
        title=os.path.basename(file_path),
        content=content,
        metadata={
            "source": file_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    # Create chunking service
    chunking_service = MarkdownChunkingService()

    # Generate chunks
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Chunking document..."),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Chunking", total=1)
        chunks = await chunking_service.chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        progress.update(task, completed=1)

    console.print(f"[green]Created {len(chunks)} chunks[/green]")

    # Create services for enrichment
    llm_service = OllamaLLMService(
        settings=settings.llm,
    )

    # Create image description service
    image_service = OllamaImageDescriptionService(
        settings=settings.llm,
        model=image_model,
        min_image_size=min_image_size,
    )

    # Create enrichment service
    enrichment_service = ImageEnhancedEnrichmentService(
        llm_service=llm_service,
        image_description_service=image_service,
        base_image_dir=base_image_dir,
    )

    # Enrich chunks
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Enriching chunks..."),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Enriching", total=len(chunks))

        # Process chunks sequentially to avoid rate limits
        enriched_chunks = []
        for chunk in chunks:
            enriched_chunk = await enrichment_service.enrich_chunk(document, chunk)
            enriched_chunks.append(enriched_chunk)
            progress.update(task, advance=1)

    return document, enriched_chunks


async def main():
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Show full enriched chunks for a document file"
    )
    parser.add_argument("document", type=str, help="Path to markdown document file")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=settings.images_dir,
        help=f"Directory containing images (default: {settings.images_dir})",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma3:12b",
        help="Model to use for text enrichment",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="gemma3:27b",
        help="Model to use for image description",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=64,
        help="Minimum size of images to process",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Size of chunks to generate"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Overlap between chunks"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "markdown", "rich"],
        default="rich",
        help="Output format",
    )

    args = parser.parse_args()

    console = Console()

    console.print(
        Panel.fit(
            f"[bold blue]Document Chunk Enrichment Viewer[/bold blue]\n"
            f"Document: [green]{args.document}[/green]\n"
            f"Image dir: [cyan]{args.image_dir}[/cyan]",
            border_style="blue",
        )
    )

    try:
        document, enriched_chunks = await process_document(
            file_path=args.document,
            base_image_dir=args.image_dir,
            llm_model=args.llm_model,
            image_model=args.image_model,
            min_image_size=args.min_image_size,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            enhance_images=True,
        )

        console.print(f"\n[bold green]Document:[/bold green] {document.title}")
        console.print(f"[bold green]Total chunks:[/bold green] {len(enriched_chunks)}")

        # Count chunks with images
        image_chunks = sum(
            1 for chunk in enriched_chunks if chunk.metadata.get("has_images", False)
        )
        console.print(f"[bold green]Chunks with images:[/bold green] {image_chunks}")

        # Display each chunk
        for i, chunk in enumerate(enriched_chunks):
            console.print(
                f"\n[bold cyan]Chunk {i+1}/{len(enriched_chunks)}[/bold cyan]"
            )

            # Check if chunk contains images
            has_images = chunk.metadata.get("has_images", False)
            image_count = chunk.metadata.get("image_count", 0)
            image_descriptions = chunk.metadata.get("image_descriptions", [])
            image_refs = chunk.metadata.get("image_refs", [])

            # Split content into sections
            content_parts = chunk.content.split("\n\n", 1)
            if len(content_parts) == 2:
                context, original = content_parts

                # Check if context contains image descriptions
                image_desc_sections = []
                general_context_sections = []

                for line in context.split("\n"):
                    if line.startswith("Image:"):
                        image_desc_sections.append(line)
                    else:
                        general_context_sections.append(line)

                general_context = "\n".join(general_context_sections).strip()
                image_context = "\n".join(image_desc_sections).strip()

                # Create panel for chunk details
                table = Table(show_header=False, box=None)
                table.add_column("Property", style="bold blue")
                table.add_column("Value")

                table.add_row("Chunk ID", chunk.id)
                table.add_row("Has Images", "✅" if has_images else "❌")
                if has_images:
                    table.add_row("Image Count", str(image_count))

                console.print(table)

                # Print general context
                console.print(
                    Panel(
                        general_context,
                        title="[bold yellow]General Context[/bold yellow]",
                        border_style="yellow",
                    )
                )

                # Print image descriptions if available
                if image_desc_sections or image_descriptions:
                    # If we have image descriptions in metadata, use those
                    if image_descriptions:
                        image_context = "\n".join(image_descriptions)
                        console.print(
                            Panel(
                                image_context,
                                title=f"[bold green]Image Descriptions (from metadata: {len(image_descriptions)})[/bold green]",
                                border_style="green",
                            )
                        )
                    # Otherwise use the descriptions parsed from context
                    elif image_desc_sections:
                        console.print(
                            Panel(
                                image_context,
                                title=f"[bold green]Image Descriptions (from context: {len(image_desc_sections)})[/bold green]",
                                border_style="green",
                            )
                        )

                    # Show image references (either from metadata or extracted from content)
                    image_refs_to_show = (
                        image_refs
                        if image_refs
                        else re.findall(r"!\[\]?\(([^)]+)\)", original)
                    )
                    if image_refs_to_show:
                        image_table = Table(title="Images Referenced", show_header=True)
                        image_table.add_column("Image Path", style="cyan")
                        image_table.add_column("Full Path")

                        for img_ref in image_refs_to_show:
                            full_path = os.path.join(args.image_dir, img_ref)
                            image_table.add_row(img_ref, full_path)

                        console.print(image_table)

                # Print original content
                if args.output_format == "markdown":
                    console.print(
                        Panel(
                            Markdown(original),
                            title="[bold cyan]Original Content[/bold cyan]",
                            border_style="cyan",
                            expand=False,
                        )
                    )
                else:
                    console.print(
                        Panel(
                            original,
                            title="[bold cyan]Original Content[/bold cyan]",
                            border_style="cyan",
                            expand=False,
                        )
                    )
            else:
                # If no splitting occurred, just show the whole content
                console.print(
                    Panel(
                        chunk.content,
                        title="[bold cyan]Chunk Content[/bold cyan]",
                        border_style="cyan",
                    )
                )

            # Print divider
            console.print("─" * 80)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
