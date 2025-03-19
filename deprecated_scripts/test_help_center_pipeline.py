#!/usr/bin/env python3
"""
Quick test script for the help center pipeline components.
"""

import os
import typer
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

from epic_rag.domain.models.document import Document
from epic_rag.infrastructure.container import setup_container, container
from epic_rag.application.use_cases.ingest_document import IngestDocumentUseCase
from html2md import convert_html_to_markdown

app = typer.Typer(add_completion=False)
console = Console()


def main(
    limit: int = typer.Option(5, "--limit", "-l", help="Number of pages to process"),
    apply_enrichment: bool = typer.Option(
        True, "--enrichment/--no-enrichment", help="Apply enrichment"
    ),
):
    """Test processing a small number of help center pages using the components."""
    console.print(
        Panel(
            f"[bold]Testing Help Center Pipeline Components[/bold]\n\n"
            f"Processing {limit} pages with enrichment {'enabled' if apply_enrichment else 'disabled'}",
            title="Pipeline Test",
            border_style="green",
        )
    )

    # Initialize container
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

    json_path = "output/epic-docs.json"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        pages = data.get("pages", [])
        total_pages = len(pages)

        # Apply limit
        if limit > 0 and limit < total_pages:
            pages = pages[:limit]

        console.print(f"Loaded {len(pages)} of {total_pages} pages from JSON")

        # Step 2: Convert to markdown
        console.print("\n[bold blue]Step 2:[/bold blue] Converting pages to markdown")

        # Create output directory
        output_dir = "data/test_help_center"
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
            task = progress.add_task("Converting pages", total=len(pages))

            for i, page in enumerate(pages):
                try:
                    title = page.get("title", f"Untitled_Page_{i}")
                    raw_html = page.get("html", "")

                    # Clean title for filename
                    clean_title = (
                        title.replace(" ", "_")
                        .replace("/", "-")
                        .replace(":", "")
                        .lower()
                    )

                    # Convert to markdown
                    markdown = convert_html_to_markdown(
                        raw_html,
                        images_dir="output/images",
                        heading_style="ATX",
                        wrap=True,
                    )

                    # Save markdown
                    md_path = os.path.join(output_dir, "markdown", f"{clean_title}.md")
                    with open(md_path, "w") as f:
                        f.write(markdown)

                    markdown_files.append((title, md_path))

                except Exception as e:
                    console.print(f"[red]Error converting page {i}: {str(e)}[/red]")

                progress.update(task, advance=1)

        console.print(f"Converted {len(markdown_files)} pages to markdown")

        # Step 3: Ingest documents
        console.print("\n[bold blue]Step 3:[/bold blue] Ingesting documents")

        # Process documents
        async def process_documents():
            results = []

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
                                "filename": os.path.basename(path),
                                "source": "epic_help_center_test",
                            },
                        )

                        # Process document
                        await use_case.execute(
                            document=document,
                            dynamic_chunking=True,
                            min_chunk_size=300,
                            max_chunk_size=800,
                            chunk_overlap=50,
                            apply_enrichment=apply_enrichment,
                        )

                        results.append(document)

                    except Exception as e:
                        console.print(f"[red]Error processing {title}: {str(e)}[/red]")

                    progress.update(task, advance=1)

            return results

        # Run processing
        documents = asyncio.run(process_documents())

        # Print results
        console.print("\n[bold green]Test completed successfully![/bold green]")
        console.print(f"Processed {len(documents)} documents")

        # Calculate statistics
        total_chunks = sum(doc.chunk_count for doc in documents)
        console.print(f"Total chunks: {total_chunks}")
        if documents:
            console.print(
                f"Average chunks per document: {total_chunks / len(documents):.2f}"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    typer.run(main)
