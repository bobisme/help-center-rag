"""Help center processing commands."""

import os
import json
import asyncio
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
from .....domain.models.document import Document
from .....infrastructure.config.settings import settings
from .....infrastructure.container import container, setup_container
from .....application.use_cases.ingest_document import IngestDocumentUseCase
from ...common import console

# Create Typer app for help center commands
help_center_app = typer.Typer()


def initialize_directories(output_dir: str) -> None:
    """Initialize the output directories.

    Args:
        output_dir: The base output directory
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "markdown"), exist_ok=True)


async def process_help_center_page(
    input_file: str,
    output_dir: str,
    convert_to_markdown: bool = True,
    save_intermediate: bool = False,
) -> Dict[str, Any]:
    """Process a help center page from JSON to markdown.

    Args:
        input_file: Path to the input JSON file
        output_dir: Path to the output directory
        convert_to_markdown: Whether to convert HTML to markdown
        save_intermediate: Whether to save intermediate files

    Returns:
        Dict containing the processed document data
    """
    # Load the JSON file
    data = load_from_json_file(input_file)

    # Extract metadata
    title = data.get("title", "Untitled")
    page_id = data.get("id", None)
    category = data.get("category", {}).get("name", "Uncategorized")
    updated_at = data.get("updated_at", None)

    # Process HTML content if present
    content = data.get("body", "")
    content_html = content

    if convert_to_markdown:
        # Preprocess the HTML
        content = preprocess_html(content)

        # Convert HTML to markdown
        content = convert_html_to_markdown(content)

    # Create output paths
    base_filename = f"{page_id}_{title.replace(' ', '_')}"[:100]
    markdown_path = os.path.join(output_dir, "markdown", f"{base_filename}.md")

    # Save markdown file
    if convert_to_markdown and save_intermediate:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(content)

    # Return the document data
    return {
        "title": title,
        "id": page_id,
        "category": category,
        "updated_at": updated_at,
        "content": f"# {title}\n\n{content}",
        "content_html": content_html,
        "markdown_path": markdown_path,
    }


@help_center_app.command("process")
def process_help_center(
    input_dir: str = typer.Option(
        "data/help_center_raw",
        "--input-dir",
        "-i",
        help="Directory containing raw JSON files",
    ),
    output_dir: str = typer.Option(
        "data/help_center",
        "--output-dir",
        "-o",
        help="Directory to output processed files",
    ),
    pattern: str = typer.Option(
        "*.json", "--pattern", "-p", help="Glob pattern to match JSON files"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Maximum number of files to process"
    ),
    save_files: bool = typer.Option(
        True,
        "--save-files/--no-save-files",
        help="Whether to save intermediate markdown files",
    ),
    ingest: bool = typer.Option(
        True,
        "--ingest/--no-ingest",
        help="Whether to ingest documents into the RAG system",
    ),
):
    """Process help center documentation from JSON to markdown and ingest into the RAG system."""
    # Initialize directories
    initialize_directories(output_dir)

    # Find input files
    input_files = glob.glob(os.path.join(input_dir, pattern))
    if not input_files:
        console.print(
            f"[bold red]No files found matching pattern {pattern} in {input_dir}[/bold red]"
        )
        return

    # Sort files by name
    input_files.sort()

    # Limit the number of files if requested
    if limit is not None:
        input_files = input_files[:limit]

    console.print(
        Panel(
            f"[bold]Processing {len(input_files)} Help Center Documents[/bold]\n\n"
            f"Input Directory: {input_dir}\n"
            f"Output Directory: {output_dir}\n"
            f"Ingest into RAG: {'Yes' if ingest else 'No'}\n"
            f"Save Markdown Files: {'Yes' if save_files else 'No'}",
            title="Help Center Processor",
            border_style="green",
        )
    )

    # Initialize container for RAG system
    if ingest:
        setup_container()
        ingest_use_case = container.resolve(IngestDocumentUseCase)

    # Define the processing function
    async def process_files():
        processed_docs = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[status]}"),
        ) as progress:
            process_task = progress.add_task(
                "Processing documents", total=len(input_files), status=""
            )

            for i, input_file in enumerate(input_files):
                filename = os.path.basename(input_file)
                progress.update(
                    process_task, advance=0, status=f"Processing {filename}"
                )

                try:
                    # Process the file
                    doc_data = await process_help_center_page(
                        input_file, output_dir, True, save_files
                    )
                    processed_docs.append(doc_data)

                    # Ingest the document into the RAG system
                    if ingest:
                        # Create a document
                        document = Document(
                            title=doc_data["title"],
                            content=doc_data["content"],
                            source_path=input_file,
                            epic_page_id=doc_data["id"],
                            metadata={
                                "category": doc_data["category"],
                                "updated_at": doc_data["updated_at"],
                            },
                        )

                        # Ingest the document
                        await ingest_use_case.execute(
                            document,
                            {
                                "dynamic_chunking": True,
                                "min_chunk_size": 300,
                                "max_chunk_size": 800,
                                "chunk_overlap": 50,
                            },
                        )

                    progress.update(process_task, advance=1)
                except Exception as e:
                    console.print(
                        f"[bold red]Error processing {filename}: {str(e)}[/bold red]"
                    )
                    progress.update(process_task, advance=1)

        return processed_docs

    # Process the files
    results = asyncio.run(process_files())

    # Display summary
    if results:
        console.print(
            f"\n[bold green]Successfully processed {len(results)} documents[/bold green]"
        )

        # Display categories
        categories = {}
        for doc in results:
            cat = doc.get("category", "Uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        table = Table(title="Document Categories")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="green")

        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            table.add_row(cat, str(count))

        console.print(table)


@help_center_app.command("list")
def list_help_center_docs(
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of documents to list"
    )
):
    """List help center documents in the database."""
    # Initialize container
    setup_container()

    async def list_documents():
        document_repository = container.get("document_repository")

        # List documents
        docs = await document_repository.list_documents(limit=limit)
        return docs

    # Get the documents
    docs = asyncio.run(list_documents())

    if not docs:
        console.print(
            "[bold yellow]No help center documents found in the database[/bold yellow]"
        )
        return

    # Display the documents
    table = Table(title=f"Help Center Documents ({len(docs)} total)")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Epic Page ID", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Created", style="magenta")

    for doc in docs:
        category = (
            doc.metadata.get("category", "N/A") if hasattr(doc, "metadata") else "N/A"
        )
        table.add_row(
            doc.id,
            doc.title,
            doc.epic_page_id or "N/A",
            category,
            (
                doc.created_at.strftime("%Y-%m-%d")
                if hasattr(doc, "created_at")
                else "N/A"
            ),
        )

    console.print(table)


@help_center_app.command("pipeline")
def run_help_center_pipeline(
    output_dir: str = typer.Option(
        "data/help_center", "--output-dir", "-o", help="Output directory"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of documents to process"
    ),
    apply_enrichment: bool = typer.Option(
        True, "--apply-enrichment/--no-enrichment", help="Apply contextual enrichment"
    ),
):
    """Run the help center processing pipeline."""
    # Initialize container
    setup_container()

    console.print(
        Panel(
            f"[bold]Running Help Center Pipeline[/bold]\n\n"
            f"Output Directory: {output_dir}\n"
            f"Limit: {limit or 'None'}\n"
            f"Apply Enrichment: {'Yes' if apply_enrichment else 'No'}",
            title="Help Center Pipeline",
            border_style="green",
        )
    )

    # Get required services
    ingest_use_case = container.resolve(IngestDocumentUseCase)

    async def run_pipeline():
        from epic_rag.domain.services.contextual_enrichment_service import (
            ContextualEnrichmentService,
        )

        # Get contextual enrichment service if needed
        enrichment_service = None
        if apply_enrichment:
            enrichment_service = container.resolve(ContextualEnrichmentService)

        # Find input files
        input_dir = "data/help_center_raw"
        input_files = glob.glob(os.path.join(input_dir, "*.json"))

        if not input_files:
            console.print(f"[bold red]No JSON files found in {input_dir}[/bold red]")
            return

        # Sort files
        input_files.sort()

        # Limit if requested
        if limit:
            input_files = input_files[:limit]

        console.print(f"Processing {len(input_files)} help center documents")

        # Process each file
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[status]}"),
        ) as progress:
            process_task = progress.add_task(
                "Processing documents", total=len(input_files), status=""
            )

            for input_file in input_files:
                filename = os.path.basename(input_file)
                progress.update(
                    process_task, advance=0, status=f"Processing {filename}"
                )

                try:
                    # Process help center page
                    doc_data = await process_help_center_page(
                        input_file, output_dir, True, True
                    )

                    # Create document
                    document = Document(
                        title=doc_data["title"],
                        content=doc_data["content"],
                        source_path=input_file,
                        epic_page_id=doc_data["id"],
                        metadata={
                            "category": doc_data["category"],
                            "updated_at": doc_data["updated_at"],
                        },
                    )

                    # Ingest document with options
                    chunking_options = {
                        "dynamic_chunking": True,
                        "min_chunk_size": 300,
                        "max_chunk_size": 800,
                        "chunk_overlap": 50,
                        "apply_enrichment": apply_enrichment,
                    }

                    await ingest_use_case.execute(document, chunking_options)

                    progress.update(process_task, advance=1)

                except Exception as e:
                    console.print(
                        f"[bold red]Error processing {filename}: {str(e)}[/bold red]"
                    )
                    progress.update(process_task, advance=1)

        console.print("[bold green]Pipeline completed successfully[/bold green]")

    # Run the pipeline
    asyncio.run(run_pipeline())


def register_commands(app: typer.Typer):
    """Register help center commands with the main app."""
    app.add_typer(
        help_center_app, name="help-center", help="Help center processing commands"
    )
