#!/usr/bin/env python
"""Utility to display all chunks for a specific document title."""

import asyncio
import json
import argparse
import sqlite3
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from epic_rag.infrastructure.config.settings import settings
from epic_rag.infrastructure.container import setup_container

# Initialize rich console
console = Console()


def get_documents_by_title(title: str):
    """Get all documents matching the given title from the database."""
    try:
        conn = sqlite3.connect("data/epic_rag.db")
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
        conn.close()


def get_chunks_for_document(document_id: str):
    """Get all chunks for a specific document ID."""
    try:
        conn = sqlite3.connect("data/epic_rag.db")
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
        conn.close()


def display_document_chunks(
    document_title: str,
    show_metadata: bool = False,
    show_context_only: bool = False,
    limit_chunks: int = None,
    auto_select_first: bool = True,
):
    """Display all chunks for documents matching the given title."""
    # Get documents matching the title
    documents = get_documents_by_title(document_title)

    if not documents:
        console.print(
            f"[bold red]No documents found with title containing '{document_title}'[/bold red]"
        )
        return

    # Display document options if multiple found
    if len(documents) > 1:
        console.print(
            f"[bold]Found {len(documents)} documents matching '{document_title}':[/bold]"
        )

        table = Table(title="Matching Documents")
        table.add_column("Index", style="cyan")
        table.add_column("Document ID", style="dim")
        table.add_column("Title", style="green")

        for i, doc in enumerate(documents):
            table.add_row(str(i + 1), doc["id"], doc["title"])

        console.print(table)

        # Auto-select first document if requested
        if auto_select_first:
            selected_doc = documents[0]
            console.print(
                f"[bold yellow]Auto-selecting first document: {selected_doc['title']}[/bold yellow]"
            )
        else:
            # Ask user to select a document
            try:
                selected_index = console.input(
                    "[bold]Enter the index of the document to view (or 'q' to quit): [/bold]"
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
    if limit_chunks:
        chunks = chunks[:limit_chunks]

    # Display document info
    console.print()
    console.print(f"[bold]Document:[/bold] {selected_doc['title']}")
    console.print(f"[bold]ID:[/bold] {selected_doc['id']}")
    console.print(f"[bold]Total Chunks:[/bold] {len(chunks)}")
    console.print()

    # Display chunks
    for i, chunk in enumerate(chunks):
        metadata = json.loads(chunk["metadata"]) if chunk["metadata"] else {}

        # Check if chunk is enriched
        is_enriched = metadata.get("enriched", False)
        context = metadata.get("context", "No context available")

        # Skip to next chunk if showing only context and chunk is not enriched
        if show_context_only and not is_enriched:
            continue

        # Prepare content display
        content = chunk["content"]

        # If showing only context, display just that
        if show_context_only:
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
                Markdown(display_content) if not show_context_only else display_content,
                title=panel_title,
                border_style="blue",
                expand=False,
            )
        )

        # Show metadata if requested
        if show_metadata:
            # Filter out embedding and other large fields
            filtered_metadata = {
                k: v
                for k, v in metadata.items()
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


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Display chunks for a specific document"
    )
    parser.add_argument("title", help="Document title to search for")
    parser.add_argument(
        "--metadata", "-m", action="store_true", help="Show chunk metadata"
    )
    parser.add_argument(
        "--context-only",
        "-c",
        action="store_true",
        help="Show only context from enriched chunks",
    )
    parser.add_argument(
        "--limit", "-l", type=int, help="Limit number of chunks to display"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enable interactive document selection",
    )

    args = parser.parse_args()

    # Initialize container
    setup_container()

    # Display document chunks
    display_document_chunks(
        document_title=args.title,
        show_metadata=args.metadata,
        show_context_only=args.context_only,
        limit_chunks=args.limit,
        auto_select_first=not args.interactive,
    )


if __name__ == "__main__":
    main()
