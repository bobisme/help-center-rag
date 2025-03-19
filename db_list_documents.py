#!/usr/bin/env python3
import asyncio
import os
import sys
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.getcwd())
from epic_rag.infrastructure.container import container, setup_container

setup_container()

console = Console()


async def main():
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="List documents in the database")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum number of documents to list",
    )
    parser.add_argument(
        "--offset", "-o", type=int, default=0, help="Offset to start listing from"
    )
    args = parser.parse_args()

    document_repository = container.get("document_repository")
    print("Fetching documents...")
    documents = await document_repository.list_documents(
        limit=args.limit, offset=args.offset
    )

    if not documents:
        print("No documents found in the database")
        return

    # Create a table
    table = Table(title=f"Documents ({args.offset+1}-{args.offset+len(documents)})")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Epic Page ID", style="green")
    table.add_column("Chunks", style="blue")
    table.add_column("Created", style="magenta")
    table.add_column("Updated", style="yellow")

    for doc in documents:
        chunks = await document_repository.get_document_chunks(doc.id)
        table.add_row(
            doc.id,
            doc.title,
            doc.epic_page_id or "N/A",
            str(len(chunks)),
            doc.created_at.strftime("%Y-%m-%d %H:%M"),
            doc.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)

    # Show total count
    stats = await document_repository.get_statistics()
    total_docs = stats.get("document_count", {}).get("value", 0)
    if total_docs > len(documents):
        print(f"Showing {len(documents)} of {total_docs} total documents")
        print(f"Use --offset to see more documents")


if __name__ == "__main__":
    asyncio.run(main())
