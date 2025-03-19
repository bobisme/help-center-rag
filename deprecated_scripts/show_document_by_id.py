#!/usr/bin/env python
"""Utility to display document chunks by document ID."""

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


def get_document_by_id(document_id: str):
    """Get document by ID from the database."""
    try:
        conn = sqlite3.connect("data/epic_rag.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get document by ID
        cursor.execute("SELECT id, title FROM documents WHERE id = ?", (document_id,))
        document = cursor.fetchone()

        return document
    except Exception as e:
        console.print(f"[bold red]Error querying database:[/bold red] {str(e)}")
        return None
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
    document_id: str,
    show_metadata: bool = False,
    show_context_only: bool = False,
    limit_chunks: int = None,
):
    """Display all chunks for a specific document ID."""
    # Get document by ID
    document = get_document_by_id(document_id)

    if not document:
        console.print(f"[bold red]No document found with ID '{document_id}'[/bold red]")
        return

    # Get chunks for the document
    chunks = get_chunks_for_document(document_id)

    if not chunks:
        console.print(
            f"[bold yellow]No chunks found for document '{document['title']}'[/bold yellow]"
        )
        return

    # Limit chunks if specified
    if limit_chunks:
        chunks = chunks[:limit_chunks]

    # Display document info
    console.print()
    console.print(f"[bold]Document:[/bold] {document['title']}")
    console.print(f"[bold]ID:[/bold] {document_id}")
    console.print(f"[bold]Total Chunks:[/bold] {len(chunks)}")
    console.print()

    # Filter chunks to only show those with context if requested
    if show_context_only:
        enriched_chunks = []
        for chunk in chunks:
            metadata = json.loads(chunk["metadata"]) if chunk["metadata"] else {}
            if metadata.get("enriched", False):
                enriched_chunks.append(chunk)

        if not enriched_chunks:
            console.print(
                "[bold yellow]No enriched chunks found for this document[/bold yellow]"
            )
            return

        chunks = enriched_chunks

    # Display chunks
    for i, chunk in enumerate(chunks):
        metadata = json.loads(chunk["metadata"]) if chunk["metadata"] else {}

        # Check if chunk is enriched
        is_enriched = metadata.get("enriched", False)
        context = metadata.get("context", "No context available")

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
    parser = argparse.ArgumentParser(description="Display chunks for a document by ID")
    parser.add_argument("id", help="Document ID to search for")
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

    args = parser.parse_args()

    # Initialize container
    setup_container()

    # Display document chunks
    display_document_chunks(
        document_id=args.id,
        show_metadata=args.metadata,
        show_context_only=args.context_only,
        limit_chunks=args.limit,
    )


if __name__ == "__main__":
    main()
