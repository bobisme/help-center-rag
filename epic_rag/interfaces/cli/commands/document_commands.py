"""Document-related CLI commands."""

import asyncio
import glob
import os
import typer
from pathlib import Path
from typing import Optional

from ....domain.models.document import Document
from ....infrastructure.container import container
from ....application.use_cases.ingest_document import IngestDocumentUseCase
from .common import console, create_progress_bar, display_document_info

document_app = typer.Typer(pretty_exceptions_enable=False)


@document_app.command("ingest")
def ingest_documents(
    source_dir: str = typer.Option(
        ..., "--source-dir", "-s", help="Directory containing markdown files"
    ),
    pattern: str = typer.Option(
        "*.md", "--pattern", "-p", help="Pattern to match markdown files"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of files to process"
    ),
    dynamic_chunking: bool = typer.Option(
        True,
        "--dynamic-chunking/--fixed-chunking",
        help="Use dynamic chunking based on content structure",
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
    """Ingest documents from a directory into the system."""
    # Find all markdown files matching the pattern
    file_paths = glob.glob(os.path.join(source_dir, pattern))

    # Sort by modification time (newest first)
    file_paths = sorted(file_paths, key=os.path.getmtime, reverse=True)

    if limit:
        file_paths = file_paths[:limit]

    console.print(f"Found [bold]{len(file_paths)}[/bold] documents to process")

    # Get the ingest use case from the container
    ingest_use_case = container.get("ingest_document_use_case")

    with create_progress_bar() as progress:
        # Add a task to the progress bar
        task = progress.add_task(
            "Processing documents", total=len(file_paths), status=""
        )

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            progress.update(task, advance=0, status=f"Processing {file_name}")

            # Read the document content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract the document title from the first heading or filename
            lines = content.split("\n")
            title = None
            for line in lines:
                if line.startswith("# "):
                    title = line.replace("# ", "").strip()
                    break

            if not title:
                title = Path(file_path).stem.replace("-", " ").title()

            # Create the document
            document = Document(
                title=title,
                content=content,
                source_path=file_path,
            )

            # Ingest the document
            asyncio.run(
                ingest_use_case.execute(
                    document=document,
                    dynamic_chunking=dynamic_chunking,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                    apply_enrichment=True,
                )
            )

            # Update the progress bar
            progress.update(task, advance=1)

    console.print("[bold green]Done![/bold green]")


@document_app.command("chunks")
def show_document_chunks(
    title: str = typer.Option(
        ..., "--title", "-t", help="Document title to display chunks for"
    ),
    show_text: bool = typer.Option(
        True, "--show-text/--hide-text", help="Show chunk text content"
    ),
    show_embeddings: bool = typer.Option(
        False, "--show-embeddings/--hide-embeddings", help="Show embedding vectors"
    ),
):
    """Show the chunks for a document with their metadata."""
    from ....domain.repositories.document_repository import DocumentRepository

    async def get_document_with_chunks():
        doc_repo = container.get("document_repository")
        document = await doc_repo.get_by_title(title)
        if not document:
            console.print(f"[bold red]Document not found: {title}[/bold red]")
            return None

        document.chunks = await doc_repo.get_chunks_for_document(document.id)
        return document

    document = asyncio.run(get_document_with_chunks())
    if not document:
        return

    display_document_info(document)

    console.print(f"\n[bold]Chunks ([cyan]{len(document.chunks)}[/cyan]):[/bold]")

    for i, chunk in enumerate(document.chunks):
        console.print(f"[bold cyan]Chunk {i+1}[/bold cyan]")
        console.print(f"ID: {chunk.id}")
        console.print(f"Sequence: {chunk.sequence}")
        console.print(f"Token Count: {chunk.token_count}")

        if chunk.metadata:
            console.print("Metadata:")
            for key, value in chunk.metadata.items():
                console.print(f"  {key}: {value}")

        if show_text:
            console.print("\nContent:")
            console.print(f"[dim]{chunk.content}[/dim]")

        if show_embeddings and hasattr(chunk, "embedding") and chunk.embedding:
            console.print("\nEmbedding:")
            embedding_preview = ", ".join([f"{v:.4f}" for v in chunk.embedding[:5]])
            console.print(f"[dim]{embedding_preview}...[/dim]")

        console.print()


def register_commands(app: typer.Typer):
    """Register document commands with the main app."""
    app.add_typer(document_app, name="documents", help="Document management commands")
