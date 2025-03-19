"""Document visualization commands for the CLI."""

import asyncio
import json
import sqlite3
import typer
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from .....infrastructure.config.settings import settings
from .....infrastructure.container import container, setup_container
from ...common import console

# Create a Typer app for document visualization commands
vis_app = typer.Typer()


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


@vis_app.command("chunks")
def show_chunks(
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
    """Display chunks for a specific document."""
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


@vis_app.command("document")
def show_document_by_id(
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
    """Register document visualization commands with the main app."""
    app.add_typer(vis_app, name="vis", help="Document visualization commands")
