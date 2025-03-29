"""Document-related CLI commands."""

import asyncio
import glob
import os
import json
import sqlite3
import typer
from pathlib import Path
from typing import Optional
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from ....domain.models.document import Document
from ....infrastructure.container import container, setup_container
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


def get_documents_by_title(title: str):
    """Get all documents matching the given title from the database."""
    conn = None
    try:
        # For SQLite, the database is typically stored in 'data/epic_rag.db'
        db_path = "data/epic_rag.db"

        # Check if the database exists
        if not os.path.exists(db_path):
            console.print(f"[bold red]Database file not found: {db_path}[/bold red]")
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all documents matching the title
        cursor.execute(
            "SELECT id, title FROM documents WHERE title LIKE ?", (f"%{title}%",)
        )
        documents = cursor.fetchall()

        return documents
    except Exception as e:
        console.print(f"[bold red]Error querying database:[/bold red] {str(e)}")
        return []
    finally:
        if conn:
            conn.close()


def get_chunks_for_document(document_id: str):
    """Get all chunks for a specific document ID."""
    conn = None
    try:
        # For SQLite, the database is typically stored in 'data/epic_rag.db'
        db_path = "data/epic_rag.db"

        # Check if the database exists
        if not os.path.exists(db_path):
            console.print(f"[bold red]Database file not found: {db_path}[/bold red]")
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all chunks for the document
        cursor.execute(
            "SELECT id, chunk_index, content, metadata FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        )
        chunks = cursor.fetchall()

        return chunks
    except Exception as e:
        console.print(f"[bold red]Error querying database:[/bold red] {str(e)}")
        return []
    finally:
        if conn:
            conn.close()


@document_app.command("search")
def search_documents(
    title: str = typer.Argument(..., help="Document title to search for"),
    metadata: bool = typer.Option(
        False, "--metadata", "-m", help="Show chunk metadata"
    ),
    context_only: bool = typer.Option(
        False, "--context-only", "-c", help="Show only context from enriched chunks"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of chunks to display"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enable interactive document selection"
    ),
):
    """Find and display documents matching a title search."""
    # Get documents matching the title
    documents = get_documents_by_title(title)

    if not documents:
        console.print(
            f"[bold red]No documents found with title containing '{title}'[/bold red]"
        )
        return

    # Display document options if multiple found
    if len(documents) > 1:
        console.print(
            f"[bold]Found {len(documents)} documents matching '{title}':[/bold]"
        )

        table = Table(title="Matching Documents")
        table.add_column("Index", style="cyan")
        table.add_column("Document ID", style="dim")
        table.add_column("Title", style="green")

        for i, doc in enumerate(documents):
            table.add_row(str(i + 1), doc["id"], doc["title"])

        console.print(table)

        # Auto-select first document if requested
        if not interactive:
            selected_doc = documents[0]
            console.print(
                f"[bold yellow]Auto-selecting first document: {selected_doc['title']}[/bold yellow]"
            )
        else:
            # Ask user to select a document
            try:
                selected_index = Prompt.ask(
                    "[bold]Enter the index of the document to view (or 'q' to quit)[/bold]"
                )
                if selected_index.lower() == "q":
                    return

                try:
                    selected_doc = documents[int(selected_index) - 1]
                except (ValueError, IndexError):
                    console.print("[bold red]Invalid selection[/bold red]")
                    return
            except EOFError:
                # Handle EOF error in non-interactive environments
                selected_doc = documents[0]
                console.print(
                    f"[bold yellow]Auto-selecting first document due to non-interactive mode: {selected_doc['title']}[/bold yellow]"
                )
    else:
        selected_doc = documents[0]

    # Get chunks for the selected document
    chunks = get_chunks_for_document(selected_doc["id"])

    if not chunks:
        console.print(
            f"[bold yellow]No chunks found for document '{selected_doc['title']}'[/bold yellow]"
        )
        return

    # Limit chunks if specified
    if limit:
        chunks = chunks[:limit]

    # Display document info
    console.print()
    console.print(f"[bold]Document:[/bold] {selected_doc['title']}")
    console.print(f"[bold]ID:[/bold] {selected_doc['id']}")
    console.print(f"[bold]Total Chunks:[/bold] {len(chunks)}")
    console.print()

    # Display chunks
    for i, chunk in enumerate(chunks):
        metadata_json = json.loads(chunk["metadata"]) if chunk["metadata"] else {}

        # Check if chunk is enriched
        is_enriched = metadata_json.get("enriched", False)
        context = metadata_json.get("context", "No context available")

        # Skip to next chunk if showing only context and chunk is not enriched
        if context_only and not is_enriched:
            continue

        # Prepare content display
        content = chunk["content"]

        # If showing only context, display just that
        if context_only:
            display_content = context
        else:
            display_content = content

        # Create panel title with enrichment indicator
        panel_title = f"Chunk {chunk['chunk_index'] + 1}"
        if is_enriched:
            panel_title += " [green](Enriched)[/green]"

        # Display the chunk content
        console.print(
            Panel(
                Markdown(display_content) if not context_only else display_content,
                title=panel_title,
                border_style="blue",
                expand=False,
            )
        )

        # Show metadata if requested
        if metadata:
            # Filter out embedding and other large fields
            filtered_metadata = {
                k: v
                for k, v in metadata_json.items()
                if k not in ["embedding", "vector_id", "image_descriptions"]
            }

            table = Table(title="Chunk Metadata")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for key, value in filtered_metadata.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                table.add_row(key, str(value))

            console.print(table)
            console.print()


@document_app.command("view")
def view_document_by_id(
    document_id: str = typer.Argument(..., help="Document ID to retrieve"),
    context_only: bool = typer.Option(
        False, "--context-only", "-c", help="Show only context from enriched chunks"
    ),
    metadata: bool = typer.Option(
        False, "--metadata", "-m", help="Show document metadata"
    ),
):
    """Show a specific document by ID."""

    async def get_document():
        """Get document from repository."""
        document_repository = container.get("document_repository")
        document = await document_repository.get_document(document_id)

        if not document:
            console.print(
                f"[bold red]Document not found with ID: {document_id}[/bold red]"
            )
            return None

        # Get chunks
        document.chunks = await document_repository.get_document_chunks(document_id)
        return document

    # Initialize container
    setup_container()

    # Get document
    document = asyncio.run(get_document())

    if not document:
        return

    # Display document info
    console.print(f"[bold]Document:[/bold] {document.title}")
    console.print(f"[bold]ID:[/bold] {document.id}")
    console.print(f"[bold]Created:[/bold] {document.created_at}")
    console.print(f"[bold]Updated:[/bold] {document.updated_at}")

    # Show metadata if requested
    if metadata and document.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in document.metadata.items():
            console.print(f"  [cyan]{key}:[/cyan] {value}")

    # Display chunks
    if hasattr(document, "chunks") and document.chunks:
        console.print(f"\n[bold]Chunks:[/bold] {len(document.chunks)} total")

        for i, chunk in enumerate(document.chunks):
            # Skip to next chunk if showing only context and chunk is not enriched
            is_enriched = (
                chunk.metadata.get("enriched", False)
                if hasattr(chunk, "metadata")
                else False
            )
            context = (
                chunk.metadata.get("context", "No context available")
                if hasattr(chunk, "metadata")
                else "No context available"
            )

            if context_only and not is_enriched:
                continue

            # Create panel title with enrichment indicator
            panel_title = f"Chunk {i + 1}"
            if is_enriched:
                panel_title += " [green](Enriched)[/green]"

            # Display appropriate content
            if context_only:
                display_content = context
            else:
                display_content = chunk.content

            console.print(
                Panel(
                    Markdown(display_content) if not context_only else display_content,
                    title=panel_title,
                    border_style="blue",
                    expand=False,
                )
            )

            # Show chunk metadata if requested
            if metadata and hasattr(chunk, "metadata") and chunk.metadata:
                filtered_metadata = {
                    k: v
                    for k, v in chunk.metadata.items()
                    if k not in ["embedding", "vector_id", "image_descriptions"]
                }

                if filtered_metadata:
                    table = Table(title="Chunk Metadata")
                    table.add_column("Key", style="cyan")
                    table.add_column("Value", style="green")

                    for key, value in filtered_metadata.items():
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        table.add_row(key, str(value))

                    console.print(table)
                    console.print()


def register_commands(app: typer.Typer):
    """Register document commands with the main app."""
    app.add_typer(document_app, name="documents", help="Document management commands")
