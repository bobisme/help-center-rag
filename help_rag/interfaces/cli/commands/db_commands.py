"""Database management CLI commands."""

import asyncio
import shutil
import subprocess
import sys
import os
import datetime
import aiosqlite
from typing import Optional
from pathlib import Path

import typer
from rich.prompt import Confirm
from rich.panel import Panel

from ....infrastructure.config.settings import settings
from ....infrastructure.container import container
from .common import console

db_app = typer.Typer(pretty_exceptions_enable=False)


@db_app.command("info")
def db_info():
    """Display information about the database."""

    async def get_db_info():
        db_path = settings.database.path

        # Check if the database exists
        if not os.path.exists(db_path):
            console.print(f"[bold red]Database file not found: {db_path}[/bold red]")
            return

        # Get the database size
        db_size = os.path.getsize(db_path)
        db_size_mb = db_size / (1024 * 1024)

        # Connect to the database
        async with aiosqlite.connect(db_path) as conn:
            # Get document count
            async with conn.execute("SELECT COUNT(*) FROM documents") as cursor:
                document_count = await cursor.fetchone()
                document_count = document_count[0] if document_count else 0

            # Get chunk count
            async with conn.execute("SELECT COUNT(*) FROM chunks") as cursor:
                chunk_count = await cursor.fetchone()
                chunk_count = chunk_count[0] if chunk_count else 0

            # Get cache count
            cache_count = 0
            try:
                async with conn.execute(
                    "SELECT COUNT(*) FROM embedding_cache"
                ) as cursor:
                    cache_count = await cursor.fetchone()
                    cache_count = cache_count[0] if cache_count else 0
            except aiosqlite.Error:
                # Table might not exist
                pass

            # Get the most recent document
            most_recent_doc = None
            try:
                async with conn.execute(
                    "SELECT id, title, created_at FROM documents ORDER BY created_at DESC LIMIT 1"
                ) as cursor:
                    most_recent_doc = await cursor.fetchone()
            except aiosqlite.Error:
                pass

        # Display the information
        console.print(f"[bold]Database Path:[/bold] {db_path}")
        console.print(f"[bold]Database Size:[/bold] {db_size_mb:.2f} MB")
        console.print(f"[bold]Document Count:[/bold] {document_count}")
        console.print(f"[bold]Chunk Count:[/bold] {chunk_count}")
        console.print(f"[bold]Embedding Cache Count:[/bold] {cache_count}")

        if most_recent_doc:
            console.print(
                f"[bold]Most Recent Document:[/bold] {most_recent_doc[1]} (ID: {most_recent_doc[0]}, Added: {most_recent_doc[2]})"
            )

    asyncio.run(get_db_info())


@db_app.command("backup")
def backup_db(
    backup_dir: str = typer.Option(
        "./backups", "--backup-dir", "-d", help="Directory to store backups"
    )
):
    """Backup the SQLite database."""
    db_path = settings.database.path

    # Create the backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)

    # Generate a backup filename with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"help_rag_backup_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_filename)

    # Copy the database file
    try:
        shutil.copy2(db_path, backup_path)
        console.print(f"[bold green]Database backed up to:[/bold green] {backup_path}")
    except Exception as e:
        console.print(f"[bold red]Error backing up database:[/bold red] {str(e)}")


@db_app.command("vacuum")
def vacuum_db():
    """Optimize the SQLite database by running VACUUM."""

    async def vacuum_database():
        db_path = settings.database.path

        # Get the database size before vacuum
        db_size_before = os.path.getsize(db_path)
        db_size_before_mb = db_size_before / (1024 * 1024)

        console.print(
            f"[bold]Database size before vacuum:[/bold] {db_size_before_mb:.2f} MB"
        )
        console.print("[bold]Running VACUUM...[/bold]")

        # Connect to the database and run VACUUM
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("VACUUM")
            await conn.commit()

        # Get the database size after vacuum
        db_size_after = os.path.getsize(db_path)
        db_size_after_mb = db_size_after / (1024 * 1024)

        space_saved = db_size_before - db_size_after
        space_saved_mb = space_saved / (1024 * 1024)

        console.print(
            f"[bold]Database size after vacuum:[/bold] {db_size_after_mb:.2f} MB"
        )
        console.print(f"[bold]Space saved:[/bold] {space_saved_mb:.2f} MB")

    asyncio.run(vacuum_database())


@db_app.command("cleanup-orphans")
def cleanup_orphans():
    """Remove orphaned chunks (chunks with no parent document)."""

    async def cleanup_orphaned_chunks():
        db_path = settings.database.path

        # Connect to the database
        async with aiosqlite.connect(db_path) as conn:
            # Find orphaned chunks
            async with conn.execute(
                """
                SELECT c.id, c.sequence
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE d.id IS NULL
                """
            ) as cursor:
                orphaned_chunks = await cursor.fetchall()
                # Convert to list to ensure we can get its length
                orphaned_chunks = list(orphaned_chunks)

            if not orphaned_chunks:
                console.print("[bold green]No orphaned chunks found[/bold green]")
                return

            # Display the orphaned chunks
            console.print(f"[bold]Found {len(orphaned_chunks)} orphaned chunks[/bold]")

            # Confirm deletion
            if Confirm.ask("Do you want to delete these orphaned chunks?"):
                # Delete the orphaned chunks
                await conn.execute(
                    """
                    DELETE FROM chunks
                    WHERE id IN (
                        SELECT c.id
                        FROM chunks c
                        LEFT JOIN documents d ON c.document_id = d.id
                        WHERE d.id IS NULL
                    )
                    """
                )
                await conn.commit()

                console.print(
                    f"[bold green]Deleted {len(orphaned_chunks)} orphaned chunks[/bold green]"
                )
            else:
                console.print("[bold yellow]Operation cancelled[/bold yellow]")

    asyncio.run(cleanup_orphaned_chunks())


@db_app.command("inspect-document")
def inspect_document(
    document_id: Optional[str] = typer.Option(
        None, "--id", help="Document ID to inspect"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Document title to inspect"
    ),
    source_id: Optional[str] = typer.Option(
        None, "--source-id", help="Source page ID to search for"
    ),
    chunks: bool = typer.Option(False, "--chunks", "-c", help="Show document chunks"),
    metadata: bool = typer.Option(
        False, "--metadata", "-m", help="Show document metadata"
    ),
):
    """Inspect a document in the database."""

    async def inspect_doc():
        if not any([document_id, title, source_id]):
            console.print(
                "[bold red]Please provide at least one of: --id, --title, or --source-id[/bold red]"
            )
            return

        # Get document repository using type-based dependency injection
        from ....domain.repositories.document_repository import DocumentRepository

        document_repository = container[DocumentRepository]
        document = None

        with console.status("[bold green]Finding document..."):
            if document_id:
                document = await document_repository.get_document(document_id)
            elif source_id:
                document = await document_repository.find_document_by_source_page_id(
                    source_id
                )
            elif title:
                # Search by title (partial match)
                filters = {"title": title}
                documents = await document_repository.list_documents(
                    limit=10, filters=filters
                )
                if documents:
                    if len(documents) > 1:
                        # Show list of matching documents
                        console.print(
                            f"[yellow]Found {len(documents)} documents matching '{title}':[/yellow]"
                        )
                        for i, doc in enumerate(documents, 1):
                            console.print(
                                f"{i}. [cyan]{doc.title}[/cyan] (ID: {doc.id})"
                            )

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
        if metadata:
            console.print("\n[bold]Metadata:[/bold]")
            if document.metadata:
                for key, value in document.metadata.items():
                    console.print(f"  [cyan]{key}:[/cyan] {value}")
            else:
                console.print("  [italic]No metadata available[/italic]")

        # Show chunks if requested
        if chunks:
            doc_chunks = await document_repository.get_document_chunks(document.id)
            console.print(f"\n[bold]Chunks:[/bold] {len(doc_chunks)} total")

            for i, chunk in enumerate(doc_chunks, 1):
                console.print(
                    f"\n[bold cyan]Chunk {i}/{len(doc_chunks)}[/bold cyan] (ID: {chunk.id})"
                )
                console.print(f"[dim]Index: {chunk.chunk_index}[/dim]")

                # Only show the first 200 characters of content with ellipsis
                content_preview = chunk.content[:200] + (
                    "..." if len(chunk.content) > 200 else ""
                )
                console.print(
                    Panel(content_preview, title=f"Content Preview", expand=False)
                )

                if metadata and chunk.metadata:
                    console.print("[bold]Chunk Metadata:[/bold]")
                    for key, value in chunk.metadata.items():
                        if (
                            key == "context"
                            and isinstance(value, str)
                            and len(value) > 100
                        ):
                            # Truncate long context values
                            console.print(f"  [cyan]{key}:[/cyan] {value[:100]}...")
                        else:
                            console.print(f"  [cyan]{key}:[/cyan] {value}")

    # Run the async function using asyncio
    asyncio.run(inspect_doc())


@db_app.command("list-documents")
def list_documents(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of documents to list"),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset for pagination"),
    sort_by: str = typer.Option(
        "created_at",
        "--sort-by",
        "-s",
        help="Sort by field (id, title, created_at, word_count)",
    ),
    descending: bool = typer.Option(
        True, "--desc/--asc", help="Sort in descending or ascending order"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", help="Search in document titles"
    ),
):
    """List documents in the database."""

    async def list_docs():
        # Get document repository using type-based dependency injection
        from ....domain.repositories.document_repository import DocumentRepository

        document_repository = container[DocumentRepository]

        # Set up filters if search is provided
        filters = {}
        if search:
            filters["title"] = search

        # Validate sort_by field
        valid_sort_fields = ["id", "title", "created_at", "updated_at", "word_count"]
        if sort_by not in valid_sort_fields:
            console.print(
                f"[bold red]Invalid sort field. Valid options are: {', '.join(valid_sort_fields)}[/bold red]"
            )
            return

        with console.status("[bold green]Retrieving documents...[/bold green]"):
            documents = await document_repository.list_documents(
                limit=limit, offset=offset, filters=filters
            )

            # Sort the documents in memory based on sort_by field
            if sort_by == "title":
                documents.sort(key=lambda d: d.title or "", reverse=descending)
            elif sort_by == "created_at":
                documents.sort(key=lambda d: d.created_at, reverse=descending)
            elif sort_by == "updated_at":
                documents.sort(key=lambda d: d.updated_at, reverse=descending)
            elif sort_by == "word_count":
                documents.sort(
                    key=lambda d: d.metadata.get("word_count", 0) if d.metadata else 0,
                    reverse=descending,
                )
            elif sort_by == "id":
                documents.sort(key=lambda d: d.id or "", reverse=descending)

        if not documents:
            console.print("[yellow]No documents found[/yellow]")
            return

        # Display document count
        console.print(f"[bold]Documents:[/bold] {len(documents)} results")
        console.print()

        # Create a table to display the documents
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        table.add_column("ID", style="dim", width=36)
        table.add_column("Title")
        table.add_column("Created", width=20)
        table.add_column("Word Count", justify="right")

        for doc in documents:
            # Format the created_at timestamp
            if isinstance(doc.created_at, str):
                created_at = doc.created_at.split(" ")[0] if doc.created_at else "N/A"
            else:
                created_at = (
                    doc.created_at.strftime("%Y-%m-%d") if doc.created_at else "N/A"
                )

            # Get the word count from metadata or use N/A
            word_count = (
                str(doc.metadata.get("word_count", "N/A")) if doc.metadata else "N/A"
            )

            table.add_row(doc.id, doc.title, created_at, word_count)

        console.print(table)

        # Show pagination info
        if len(documents) == limit:
            console.print(
                f"\n[dim]Use --offset {offset + limit} to see the next page[/dim]"
            )

    # Run the async function
    asyncio.run(list_docs())


def register_commands(app: typer.Typer):
    """Register database commands with the main app."""
    app.add_typer(db_app, name="db", help="Database management commands")
