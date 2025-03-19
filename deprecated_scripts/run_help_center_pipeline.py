#!/usr/bin/env python3
"""
Wrapper script for processing Epic help center documentation.

NOTE: This is a simplified wrapper that does not use the ZenML pipeline directly.
Instead, it uses the same pipeline steps but without ZenML's orchestration.
"""

import os
import json
import asyncio
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

from html2md import convert_html_to_markdown
from epic_rag.domain.models.document import Document
from epic_rag.infrastructure.container import setup_container, container
from epic_rag.application.use_cases.ingest_document import IngestDocumentUseCase

# Create a rich console
console = Console()

# Create the app with the help text
app = typer.Typer(
    help="Process Epic help center documentation without using ZenML directly.",
    add_completion=False,
)


def main(
    json_path: str = typer.Option(
        "output/epic-docs.json", "--input", "-i", help="Path to the input JSON file"
    ),
    output_dir: str = typer.Option(
        "data/help_center",
        "--output-dir",
        "-o",
        help="Directory to save markdown output",
    ),
    images_dir: str = typer.Option(
        "output/images", "--images-dir", "-img", help="Directory containing images"
    ),
    limit: int = typer.Option(
        0, "--limit", "-l", help="Limit the number of pages to process (0 = all)"
    ),
    start_index: int = typer.Option(
        0, "--start", "-s", help="Index to start processing from"
    ),
    min_chunk_size: int = typer.Option(
        300, "--min-chunk-size", help="Minimum chunk size when using dynamic chunking"
    ),
    max_chunk_size: int = typer.Option(
        800, "--max-chunk-size", help="Maximum chunk size when using dynamic chunking"
    ),
    chunk_overlap: int = typer.Option(
        50, "--chunk-overlap", help="Overlap between chunks"
    ),
    apply_enrichment: bool = typer.Option(
        True,
        "--apply-enrichment/--no-enrichment",
        help="Whether to apply contextual enrichment",
    ),
):
    """Process Epic help center documentation using the pipeline steps without ZenML."""
    console.print(
        Panel(
            "[bold]Epic Help Center Processing Pipeline[/bold]\n\n"
            "This will convert pages from the JSON file to markdown, chunk the documents, "
            "apply contextual enrichment, and ingest them into the RAG system.",
            title="Processing Pipeline",
            border_style="cyan",
        )
    )

    # Initialize the container
    setup_container()

    # Get required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    chunking_service = container.get("chunking_service")
    embedding_service = container.get("embedding_service")

    # Create use case
    use_case = IngestDocumentUseCase(
        document_repository=document_repository,
        vector_repository=vector_repository,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )

    # Step 1: Load data from JSON
    console.print("\n[bold blue]Step 1:[/bold blue] Loading pages from JSON")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        pages = data.get("pages", [])
        total_pages = len(pages)

        # Calculate which pages to process
        if limit <= 0 or limit > total_pages - start_index:
            end_index = total_pages
        else:
            end_index = start_index + limit

        # Get selected pages
        selected_pages = pages[start_index:end_index]
        actual_count = len(selected_pages)

        console.print(
            f"Found [bold]{total_pages}[/bold] pages in [cyan]{json_path}[/cyan]"
        )
        console.print(
            f"Processing [bold]{actual_count}[/bold] pages (from index {start_index} to {end_index - 1})"
        )

        # Step 2: Convert to markdown
        console.print("\n[bold blue]Step 2:[/bold blue] Converting pages to markdown")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "markdown"), exist_ok=True)

        # Convert pages
        markdown_files = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Converting pages", total=actual_count)

            for i, page in enumerate(selected_pages):
                try:
                    title = page.get("title", f"Untitled_Page_{i + start_index}")
                    raw_html = page.get("rawHtml", "")

                    if not raw_html:
                        progress.update(task, advance=1)
                        continue

                    # Clean title for filename
                    clean_title = (
                        title.replace(" ", "_")
                        .replace("/", "-")
                        .replace(":", "")
                        .lower()
                    )

                    # Convert to markdown
                    markdown = convert_html_to_markdown(
                        raw_html, images_dir=images_dir, heading_style="ATX", wrap=True
                    )

                    # Save markdown
                    md_path = os.path.join(output_dir, "markdown", f"{clean_title}.md")
                    with open(md_path, "w") as f:
                        f.write(markdown)

                    markdown_files.append((title, md_path))

                except Exception as e:
                    console.print(
                        f"[red]Error converting page {i + start_index}: {str(e)}[/red]"
                    )

                progress.update(task, advance=1)

        console.print(
            f"Successfully converted [green]{len(markdown_files)}[/green] of [bold]{actual_count}[/bold] pages"
        )

        # Step 3: Ingest documents
        console.print("\n[bold blue]Step 3:[/bold blue] Ingesting documents")

        # Process documents
        async def process_documents():
            results = []
            errors = []

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Processing documents", total=len(markdown_files)
                )

                for title, path in markdown_files:
                    try:
                        # Update progress description
                        filename = os.path.basename(path)
                        progress.update(task, description=f"Processing {filename}")

                        # Read content
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Create document
                        document = Document(
                            title=title,
                            content=content,
                            metadata={
                                "source_path": path,
                                "file_type": "markdown",
                                "filename": filename,
                                "source": "epic_help_center",
                            },
                        )

                        # Process document
                        result = await use_case.execute(
                            document=document,
                            dynamic_chunking=True,
                            min_chunk_size=min_chunk_size,
                            max_chunk_size=max_chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        # Apply enrichment if requested
                        if apply_enrichment and result.chunks:
                            # Get the contextual enrichment service
                            enrichment_service = container.get(
                                "contextual_enrichment_service"
                            )

                            # Enrich chunks
                            enriched_chunks = await enrichment_service.enrich_chunks(
                                document=result, chunks=result.chunks
                            )

                            # Update document with enriched chunks
                            result.chunks = enriched_chunks

                            # Re-embed and store enriched chunks
                            embedded_chunks = (
                                await embedding_service.batch_embed_chunks(
                                    result.chunks
                                )
                            )
                            vector_ids = await vector_repository.batch_store_embeddings(
                                embedded_chunks
                            )

                            # Update chunks with vector IDs
                            for i, chunk in enumerate(embedded_chunks):
                                chunk.vector_id = vector_ids[i]
                                await document_repository.save_chunk(chunk)

                        results.append(result)

                    except Exception as e:
                        console.print(f"[red]Error processing {title}: {str(e)}[/red]")
                        errors.append((path, str(e)))

                    progress.update(task, advance=1)

            return results, errors

        # Run processing
        documents, errors = asyncio.run(process_documents())

        # Print results
        console.print("\n[bold green]Pipeline completed successfully![/bold green]")
        console.print(
            f"Processed [green]{len(documents)}[/green] of [bold]{len(markdown_files)}[/bold] documents"
        )

        # Calculate statistics
        total_chunks = sum(doc.chunk_count for doc in documents)
        console.print(f"Created [bold]{total_chunks}[/bold] chunks")
        if documents:
            console.print(
                f"Average chunks per document: [cyan]{total_chunks / len(documents):.2f}[/cyan]"
            )

        # Show errors if any
        if errors:
            console.print(
                f"[bold red]{len(errors)}[/bold red] documents failed to process"
            )

            if len(errors) <= 5:
                for path, error in errors:
                    console.print(
                        f"  [red]Error processing {os.path.basename(path)}: {error}[/red]"
                    )
            else:
                for path, error in errors[:5]:
                    console.print(
                        f"  [red]Error processing {os.path.basename(path)}: {error}[/red]"
                    )
                console.print(f"  [red]...and {len(errors) - 5} more errors[/red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    typer.run(main)
