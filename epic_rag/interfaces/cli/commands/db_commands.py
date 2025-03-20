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

from ....infrastructure.config.settings import settings
from .common import console

db_app = typer.Typer(pretty_exceptions_enable=False)


@db_app.command("info")
def db_info():
    """Display information about the database."""

    async def get_db_info():
        db_path = settings.sqlite_db_path

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
    db_path = settings.sqlite_db_path

    # Create the backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)

    # Generate a backup filename with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"epic_rag_backup_{timestamp}.db"
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
        db_path = settings.sqlite_db_path

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
        db_path = settings.sqlite_db_path

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
    chunks: bool = typer.Option(False, "--chunks", "-c", help="Show document chunks"),
    metadata: bool = typer.Option(
        False, "--metadata", "-m", help="Show document metadata"
    ),
):
    """Inspect a document in the database."""
    # This command had hanging issues - use a subprocess to run the actual command
    # in a separate process to avoid event loop issues

    # Pass arguments to the subprocess
    script_path = Path(settings.project_root) / "db_inspect_document.py"
    args = [sys.executable, str(script_path)]

    if document_id:
        args.extend(["--id", document_id])
    if title:
        args.extend(["--title", title])
    if chunks:
        args.append("--chunks")
    if metadata:
        args.append("--metadata")

    # Run the subprocess
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        console.print("[bold red]Error inspecting document[/bold red]")
    except FileNotFoundError:
        console.print(
            "[bold red]Error: The helper script db_inspect_document.py was not found[/bold red]"
        )


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
):
    """List documents in the database."""
    # This command had hanging issues - use a subprocess to run the actual command
    # in a separate process to avoid event loop issues

    # Pass arguments to the subprocess
    script_path = Path(settings.project_root) / "db_list_documents.py"
    args = [sys.executable, str(script_path)]

    args.extend(["--limit", str(limit)])
    args.extend(["--offset", str(offset)])
    args.extend(["--sort-by", sort_by])

    if not descending:
        args.append("--asc")

    # Run the subprocess
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        console.print("[bold red]Error listing documents[/bold red]")
    except FileNotFoundError:
        console.print(
            "[bold red]Error: The helper script db_list_documents.py was not found[/bold red]"
        )


def register_commands(app: typer.Typer):
    """Register database commands with the main app."""
    app.add_typer(db_app, name="db", help="Database management commands")
