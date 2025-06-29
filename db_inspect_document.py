#!/usr/bin/env python3
import asyncio
import os
import sys
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, os.getcwd())
from help_rag.infrastructure.container import container, setup_container

setup_container()

console = Console()


async def main():
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a document and its metadata")
    parser.add_argument("--id", help="Document ID to inspect")
    parser.add_argument("--title", help="Document title to search for")
    parser.add_argument("--source-id", help="Source page ID to search for")
    parser.add_argument(
        "--chunks", "-c", action="store_true", help="Show document chunks"
    )
    parser.add_argument(
        "--metadata", "-m", action="store_true", help="Show detailed metadata"
    )
    args = parser.parse_args()

    if not any([args.id, args.title, args.source_id]):
        print("Please provide at least one of: --id, --title, or --source-id")
        return

    document_repository = container.get("document_repository")
    document = None

    with console.status("[bold green]Finding document..."):
        if args.id:
            document = await document_repository.get_document(args.id)
        elif args.source_id:
            document = await document_repository.find_document_by_source_page_id(
                args.source_id
            )
        elif args.title:
            # Search by title (partial match)
            filters = {"title": args.title}
            documents = await document_repository.list_documents(
                limit=10, filters=filters
            )
            if documents:
                if len(documents) > 1:
                    # Show list of matching documents
                    console.print(
                        f"[yellow]Found {len(documents)} documents matching '{args.title}':[/yellow]"
                    )
                    for i, doc in enumerate(documents, 1):
                        console.print(f"{i}. [cyan]{doc.title}[/cyan] (ID: {doc.id})")

                    # Let user choose one
                    try:
                        idx = int(input("Select document number: ")) - 1
                        if 0 <= idx < len(documents):
                            document = documents[idx]
                        else:
                            console.print("[red]Invalid selection[/red]")
                            return
                    except (ValueError, EOFError):
                        # Use first document as fallback on error
                        document = documents[0]
                else:
                    document = documents[0]

    if not document:
        console.print("[yellow]No matching document found[/yellow]")
        return

    # Display document info
    console.print(f"[bold green]Document:[/bold green] {document.title}")
    console.print(f"[bold]ID:[/bold] {document.id}")
    console.print(f"[bold]Source Page ID:[/bold] {document.source_page_id or 'N/A'}")
    console.print(f"[bold]Source Path:[/bold] {document.source_path or 'N/A'}")
    console.print(f"[bold]Created:[/bold] {document.created_at}")
    console.print(f"[bold]Updated:[/bold] {document.updated_at}")

    # Show detailed metadata if requested
    if args.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        if document.metadata:
            for key, value in document.metadata.items():
                console.print(f"  [cyan]{key}:[/cyan] {value}")
        else:
            console.print("  [italic]No metadata available[/italic]")

    # Show chunks if requested
    if args.chunks:
        chunks = await document_repository.get_document_chunks(document.id)
        console.print(f"\n[bold]Chunks:[/bold] {len(chunks)} total")

        for i, chunk in enumerate(chunks, 1):
            console.print(
                f"\n[bold cyan]Chunk {i}/{len(chunks)}[/bold cyan] (ID: {chunk.id})"
            )
            console.print(f"[dim]Index: {chunk.chunk_index}[/dim]")

            # Only show the first 200 characters of content with ellipsis
            content_preview = chunk.content[:200] + (
                "..." if len(chunk.content) > 200 else ""
            )
            console.print(
                Panel(content_preview, title=f"Content Preview", expand=False)
            )

            if args.metadata and chunk.metadata:
                console.print("[bold]Chunk Metadata:[/bold]")
                for key, value in chunk.metadata.items():
                    if key == "context" and isinstance(value, str) and len(value) > 100:
                        # Truncate long context values
                        console.print(f"  [cyan]{key}:[/cyan] {value[:100]}...")
                    else:
                        console.print(f"  [cyan]{key}:[/cyan] {value}")


if __name__ == "__main__":
    asyncio.run(main())
