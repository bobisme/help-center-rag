#!/usr/bin/env python3
"""Demo for image-enhanced contextual enrichment."""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
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
from epic_rag.infrastructure.llm.contextual_enrichment_service import (
    OllamaContextualEnrichmentService,
)
from epic_rag.infrastructure.llm.image_enhanced_enrichment_service import (
    ImageEnhancedEnrichmentService,
)


async def generate_test_chunks(
    file_path: str, chunk_size: int = 500, overlap: int = 50
) -> Tuple[Document, List[DocumentChunk]]:
    """Generate test document and chunks from a markdown file.

    Args:
        file_path: Path to the markdown file
        chunk_size: Size of chunks in characters
        overlap: Overlap between chunks in characters

    Returns:
        Tuple containing the Document and a list of DocumentChunks
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Create test document
    document = Document(
        id="test-doc-1",
        title=os.path.basename(file_path),
        content=content,
        metadata={"source": file_path},
    )

    # Create simple chunks (this is a simplified chunking strategy)
    chunks = []
    for i in range(0, len(content), chunk_size - overlap):
        chunk_content = content[i : i + chunk_size]
        # Skip chunks that are too small
        if len(chunk_content) < 100:
            continue

        chunk = DocumentChunk(
            id=f"chunk-{i // (chunk_size - overlap)}",
            document_id=document.id,
            content=chunk_content,
            metadata={"index": i // (chunk_size - overlap)},
            chunk_index=i // (chunk_size - overlap),
        )
        chunks.append(chunk)

    return document, chunks


async def main():
    """Run the image-enhanced enrichment demo."""
    parser = argparse.ArgumentParser(
        description="Demo for image-enhanced contextual enrichment"
    )
    parser.add_argument(
        "document", type=str, help="Path to a markdown document with images to enrich"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=settings.images_dir,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma3:12b",
        help="Ollama model to use for textual enrichment",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="gemma3:27b",
        help="Ollama model to use for image description",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=64,
        help="Minimum size in pixels for images to process",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Size of document chunks in characters",
    )

    args = parser.parse_args()

    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]Image-Enhanced Contextual Enrichment Demo[/bold blue]\n"
            f"Document: [green]{args.document}[/green]\n"
            f"Text Model: [cyan]{args.llm_model}[/cyan] | Image Model: [cyan]{args.image_model}[/cyan]",
            border_style="blue",
        )
    )

    if not os.path.exists(args.document):
        console.print(f"[bold red]Error:[/bold red] Document {args.document} not found")
        return

    # Create services
    llm_service = OllamaLLMService(
        settings=settings.llm,
    )

    image_service = OllamaImageDescriptionService(
        settings=settings.llm,
        model=args.image_model,
        min_image_size=args.min_image_size,
    )

    # Create standard and image-enhanced enrichment services
    standard_enrichment = OllamaContextualEnrichmentService(llm_service=llm_service)

    enhanced_enrichment = ImageEnhancedEnrichmentService(
        llm_service=llm_service,
        image_description_service=image_service,
        base_image_dir=args.image_dir,
    )

    # Generate test chunks
    console.print("Preparing document chunks...")
    document, chunks = await generate_test_chunks(
        file_path=args.document, chunk_size=args.chunk_size
    )

    console.print(
        f"Generated [bold green]{len(chunks)}[/bold green] chunks from document"
    )

    # Identify chunks with images
    image_chunks = []
    for chunk in chunks:
        if "![" in chunk.content:
            image_chunks.append(chunk)

    console.print(
        f"Found [bold green]{len(image_chunks)}[/bold green] chunks containing images"
    )

    if not image_chunks:
        console.print(
            "[bold yellow]Warning:[/bold yellow] No chunks with images found in this document"
        )
        return

    # Process chunks with both methods
    console.print(
        "\n[bold]Enriching chunks with and without image descriptions...[/bold]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        # Process image chunks only for demo purposes
        standard_task = progress.add_task(
            "Standard enrichment...", total=len(image_chunks)
        )
        enhanced_task = progress.add_task(
            "Image-enhanced enrichment...", total=len(image_chunks)
        )

        # Process in parallel
        standard_results = []
        enhanced_results = []

        for idx, chunk in enumerate(image_chunks):
            # Standard enrichment
            standard_chunk = await standard_enrichment.enrich_chunk(document, chunk)
            standard_results.append(standard_chunk)
            progress.update(standard_task, advance=1)

            # Enhanced enrichment
            try:
                enhanced_chunk = await enhanced_enrichment.enrich_chunk(document, chunk)
                enhanced_results.append(enhanced_chunk)
                # Display some debug info in verbose mode
                print(
                    f"Enhanced chunk {idx+1}: {len(enhanced_chunk.metadata.get('image_descriptions', []))} image descriptions"
                )
            except Exception as e:
                console.print(f"[red]Error enriching chunk {idx+1}: {str(e)}[/red]")
                enhanced_results.append(chunk)  # Use original as fallback

            progress.update(enhanced_task, advance=1)

    # Display results
    console.print("\n[bold green]Comparison of Enrichment Methods[/bold green]")

    for i, (std_chunk, enh_chunk) in enumerate(zip(standard_results, enhanced_results)):
        console.print(f"\n[bold cyan]Chunk {i+1}[/bold cyan]")

        # Extract just the added context part
        std_context = std_chunk.content.split("\n\n")[0]
        enh_context = enh_chunk.content.split("\n\n")[0]

        table = Table(
            show_header=True, header_style="bold blue", title=f"Chunk {i+1} Enrichment"
        )
        table.add_column("Method")
        table.add_column("Generated Context")

        table.add_row("Standard", std_context)
        table.add_row("Image-Enhanced", enh_context)

        console.print(table)

        # Show metadata
        metadata_table = Table(
            show_header=True, header_style="bold blue", title="Chunk Metadata"
        )
        metadata_table.add_column("Property")
        metadata_table.add_column("Standard")
        metadata_table.add_column("Image-Enhanced")

        for key in set(enh_chunk.metadata.keys()):
            if key in std_chunk.metadata:
                std_value = str(std_chunk.metadata.get(key, "N/A"))
                enh_value = str(enh_chunk.metadata.get(key, "N/A"))
                metadata_table.add_row(key, std_value, enh_value)

        console.print(metadata_table)

        # Show image in chunk
        if "![" in std_chunk.content:
            # Extract the image reference
            image_ref = std_chunk.content.split("![")[1].split("]")[1].strip("()")
            console.print(f"\nImage referenced: [italic]{image_ref}[/italic]")


if __name__ == "__main__":
    asyncio.run(main())
