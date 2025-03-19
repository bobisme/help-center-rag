#!/usr/bin/env python3
"""
Script to process Epic Help Center documentation from JSON to markdown,
and ingest it into the RAG system.
"""

import os
import json
import asyncio
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil

import typer
from rich.console import Console
from rich.progress import Progress, TaskProgressColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

from html2md import convert_html_to_markdown, preprocess_html
from html2md.loaders import load_from_json_file
from epic_rag.domain.models.document import Document
from epic_rag.infrastructure.config.settings import settings
from epic_rag.infrastructure.container import container, setup_container
from epic_rag.application.use_cases.ingest_document import IngestDocumentUseCase

# Create a rich console for output
console = Console()

# Create the Typer app
app = typer.Typer(
    help="Process Epic Help Center documentation from JSON to markdown, and ingest it into the RAG system.",
    add_completion=False,
)


def initialize_directories(output_dir: str) -> None:
    """Initialize the output directories.

    Args:
        output_dir: The base output directory
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "markdown"), exist_ok=True)


def get_page_count(json_path: str) -> int:
    """Get the number of pages in the JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Number of pages
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            return 0

        return len(data["pages"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


def process_page_to_markdown(
    json_path: str, page_index: int, output_dir: str, images_dir: str
) -> Optional[Tuple[str, str]]:
    """Process a single page from the JSON file to markdown.

    Args:
        json_path: Path to the JSON file
        page_index: Index of the page to process
        output_dir: Directory to save markdown output
        images_dir: Directory containing images

    Returns:
        Tuple of (title, markdown_path) if successful, None otherwise
    """
    try:
        # Load page from JSON
        raw_html, title, _ = load_from_json_file(json_path, page_index)

        if not title:
            title = f"Untitled_Page_{page_index}"

        # Clean the title to create a valid filename
        clean_title = title.replace(" ", "_").replace("/", "-").replace(":", "").lower()

        # Convert HTML to markdown
        markdown = convert_html_to_markdown(
            raw_html, images_dir=images_dir, heading_style="ATX", wrap=True
        )

        # Save markdown to file
        markdown_path = os.path.join(output_dir, "markdown", f"{clean_title}.md")
        with open(markdown_path, "w") as f:
            f.write(markdown)

        return (title, markdown_path)

    except Exception as e:
        console.print(f"[red]Error processing page {page_index}: {str(e)}[/red]")
        return None


async def ingest_document(
    doc_path: str,
    title: str,
    use_case: IngestDocumentUseCase,
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
) -> Tuple[bool, Optional[str]]:
    """Ingest a markdown document into the RAG system.

    Args:
        doc_path: Path to the markdown document
        title: Title of the document
        use_case: The ingest document use case
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        chunk_overlap: Chunk overlap

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Read the file content
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create the document
        document = Document(
            title=title,
            content=content,
            metadata={
                "source_path": doc_path,
                "file_type": "markdown",
                "filename": os.path.basename(doc_path),
                "source": "epic_help_center",
                "processed_at": datetime.now().isoformat(),
            },
        )

        # Process the document
        await use_case.execute(
            document=document,
            dynamic_chunking=True,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return (True, None)

    except Exception as e:
        return (False, str(e))


@app.command("convert")
def convert_pages(
    input_file: str = typer.Option(
        "output/epic-docs.json", "--input", "-i", help="Path to the input JSON file"
    ),
    output_dir: str = typer.Option(
        "data", "--output-dir", "-o", help="Directory to save markdown output"
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
):
    """Convert pages from the JSON file to markdown."""
    # Initialize directories
    initialize_directories(output_dir)

    # Get the number of pages
    page_count = get_page_count(input_file)
    if page_count == 0:
        console.print("[red]Error: No pages found in the JSON file[/red]")
        return

    console.print(f"Found [bold]{page_count}[/bold] pages in [cyan]{input_file}[/cyan]")

    # Calculate how many pages to process
    if limit <= 0 or limit > page_count - start_index:
        end_index = page_count
    else:
        end_index = start_index + limit

    actual_count = end_index - start_index
    console.print(
        f"Processing [bold]{actual_count}[/bold] pages (from index {start_index} to {end_index - 1})"
    )

    # Process pages
    results = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting pages", total=actual_count)

        for i in range(start_index, end_index):
            # Update progress
            progress.update(task, description=f"Converting page {i}/{end_index - 1}")

            # Process page
            result = process_page_to_markdown(
                json_path=input_file,
                page_index=i,
                output_dir=output_dir,
                images_dir=images_dir,
            )

            if result:
                results.append(result)

            # Update progress
            progress.update(task, advance=1)

    # Show results
    console.print()
    console.print(
        f"Successfully converted [bold green]{len(results)}[/bold green] of [bold]{actual_count}[/bold] pages"
    )

    # Show sample of converted files
    if results:
        table = Table(title="Sample of Converted Files")
        table.add_column("Title", style="cyan")
        table.add_column("Output Path", style="green")

        for title, path in results[:5]:  # Show first 5 results
            table.add_row(title, path)

        console.print(table)

        if len(results) > 5:
            console.print(f"...and {len(results) - 5} more files")


@app.command("ingest")
def ingest_documents(
    source_dir: str = typer.Option(
        "data/markdown",
        "--source-dir",
        "-s",
        help="Directory containing markdown files",
    ),
    pattern: str = typer.Option(
        "*.md", "--pattern", "-p", help="Pattern to match markdown files"
    ),
    limit: int = typer.Option(
        0, "--limit", "-l", help="Limit number of files to process (0 = all)"
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
):
    """Ingest markdown documents into the RAG system."""
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

    # Find all markdown files matching the pattern
    file_paths = glob.glob(os.path.join(source_dir, pattern))

    # Sort by modification time (newest first)
    file_paths = sorted(file_paths, key=os.path.getmtime, reverse=True)

    # Apply limit if specified
    if limit > 0:
        file_paths = file_paths[:limit]

    # Show summary
    console.print(
        f"Found [bold]{len(file_paths)}[/bold] files in [cyan]{source_dir}[/cyan]"
    )

    # Process files
    async def process_files():
        results = []
        errors = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents", total=len(file_paths))

            for file_path in file_paths:
                # Update progress
                filename = os.path.basename(file_path)
                progress.update(task, description=f"Processing {filename}")

                # Extract title from the filename
                title = Path(filename).stem.replace("_", " ").title()

                # Ingest document
                success, error = await ingest_document(
                    doc_path=file_path,
                    title=title,
                    use_case=use_case,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if success:
                    results.append(file_path)
                else:
                    errors.append((file_path, error))

                # Update progress
                progress.update(task, advance=1)

        return results, errors

    # Run the processing
    results, errors = asyncio.run(process_files())

    # Show results
    console.print()
    console.print(
        f"Successfully ingested [bold green]{len(results)}[/bold green] of [bold]{len(file_paths)}[/bold] documents"
    )

    if errors:
        console.print(f"[bold red]{len(errors)}[/bold red] documents failed to ingest")

        table = Table(title="Failed Documents")
        table.add_column("File", style="cyan")
        table.add_column("Error", style="red")

        for file_path, error in errors[:5]:  # Show first 5 errors
            table.add_row(os.path.basename(file_path), str(error))

        console.print(table)

        if len(errors) > 5:
            console.print(f"...and {len(errors) - 5} more errors")


@app.command("pipeline")
def run_pipeline(
    input_file: str = typer.Option(
        "output/epic-docs.json", "--input", "-i", help="Path to the input JSON file"
    ),
    output_dir: str = typer.Option(
        "data", "--output-dir", "-o", help="Directory to save markdown output"
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
):
    """Run the full pipeline: convert pages to markdown and ingest them into the RAG system."""
    console.print(
        Panel(
            "[bold]Epic Help Center Processing Pipeline[/bold]\n\n"
            "This will convert pages from the JSON file to markdown and ingest them into the RAG system.",
            title="Pipeline",
            border_style="cyan",
        )
    )

    # Step 1: Convert pages to markdown
    console.print("\n[bold]Step 1:[/bold] Converting pages to markdown\n")

    # Initialize directories
    initialize_directories(output_dir)

    # Get the number of pages
    page_count = get_page_count(input_file)
    if page_count == 0:
        console.print("[red]Error: No pages found in the JSON file[/red]")
        return

    console.print(f"Found [bold]{page_count}[/bold] pages in [cyan]{input_file}[/cyan]")

    # Calculate how many pages to process
    if limit <= 0 or limit > page_count - start_index:
        end_index = page_count
    else:
        end_index = start_index + limit

    actual_count = end_index - start_index
    console.print(
        f"Processing [bold]{actual_count}[/bold] pages (from index {start_index} to {end_index - 1})"
    )

    # Process pages
    markdown_files = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting pages", total=actual_count)

        for i in range(start_index, end_index):
            # Update progress
            progress.update(task, description=f"Converting page {i}/{end_index - 1}")

            # Process page
            result = process_page_to_markdown(
                json_path=input_file,
                page_index=i,
                output_dir=output_dir,
                images_dir=images_dir,
            )

            if result:
                title, path = result
                markdown_files.append((title, path))

            # Update progress
            progress.update(task, advance=1)

    # Show results
    console.print()
    console.print(
        f"Successfully converted [bold green]{len(markdown_files)}[/bold green] of [bold]{actual_count}[/bold] pages"
    )

    # Step 2: Initialize the container for ingestion
    console.print("\n[bold]Step 2:[/bold] Ingesting documents into the RAG system\n")

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

    # Process files
    async def process_files():
        results = []
        errors = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents", total=len(markdown_files))

            for title, file_path in markdown_files:
                # Update progress
                filename = os.path.basename(file_path)
                progress.update(task, description=f"Processing {filename}")

                # Ingest document
                success, error = await ingest_document(
                    doc_path=file_path,
                    title=title,
                    use_case=use_case,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if success:
                    results.append(file_path)
                else:
                    errors.append((file_path, error))

                # Update progress
                progress.update(task, advance=1)

        return results, errors

    # Run the processing
    results, errors = asyncio.run(process_files())

    # Show results
    console.print()
    console.print(
        f"Successfully ingested [bold green]{len(results)}[/bold green] of [bold]{len(markdown_files)}[/bold] documents"
    )

    if errors:
        console.print(f"[bold red]{len(errors)}[/bold red] documents failed to ingest")

        table = Table(title="Failed Documents")
        table.add_column("File", style="cyan")
        table.add_column("Error", style="red")

        for file_path, error in errors[:5]:  # Show first 5 errors
            table.add_row(os.path.basename(file_path), str(error))

        console.print(table)

        if len(errors) > 5:
            console.print(f"...and {len(errors) - 5} more errors")

    # Summary
    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    console.print(
        f"Converted [bold]{len(markdown_files)}[/bold] pages and ingested [bold]{len(results)}[/bold] documents"
    )


@app.command("list")
def list_pages(
    input_file: str = typer.Option(
        "output/epic-docs.json", "--input", "-i", help="Path to the input JSON file"
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Limit the number of pages to list (0 = all)"
    ),
):
    """List pages in the JSON file."""
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            console.print(f"[red]Error: Invalid JSON structure in {input_file}[/red]")
            return

        pages = data["pages"]
        page_count = len(pages)

        console.print(f"Available pages in [cyan]{input_file}[/cyan]:")
        console.print(f"Total pages: [green]{page_count}[/green]")

        # Create a table
        table = Table(show_header=True, header_style="bold")
        table.add_column("INDEX", style="dim", width=8)
        table.add_column("TITLE", style="cyan")

        # Decide how many pages to show
        if limit <= 0 or limit > page_count:
            display_pages = pages
        else:
            display_pages = pages[:limit]

        # Add rows to the table
        for i, page in enumerate(display_pages):
            title = page.get("title", "Untitled")
            table.add_row(str(i), title)

        console.print(table)

        if limit > 0 and limit < page_count:
            console.print(
                f"Showing {limit} of {page_count} pages. Use --limit 0 to see all."
            )

    except FileNotFoundError:
        console.print(f"[red]Error: File {input_file} not found[/red]")
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON format in {input_file}[/red]")


if __name__ == "__main__":
    app()
