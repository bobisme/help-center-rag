"""Utility CLI commands."""

import asyncio
import os

import typer
from rich.table import Table

from ....infrastructure.config.settings import settings
from ....infrastructure.container import container
from .common import console

utility_app = typer.Typer(pretty_exceptions_enable=False)


@utility_app.command("info")
def show_info():
    """Display system information."""
    # Display system settings
    console.print("[bold]System Information:[/bold]")
    console.print(f"Project Root: {settings.project_root}")
    console.print(f"SQLite DB Path: {settings.database.path}")
    console.print(f"Vector Store URL: {settings.qdrant.url or 'Local'}")
    console.print(f"Vector Store Collection: {settings.qdrant.collection_name}")
    console.print(f"Embedding Provider: {settings.embedding.provider}")
    console.print(f"Embedding Model: {settings.embedding.model}")
    console.print(f"LLM Provider: {settings.llm.provider}")
    console.print(f"LLM Model: {settings.llm.model}")
    console.print(
        f"Reranker Model: {settings.retrieval.reranker.model_name if settings.retrieval.reranker.enabled else 'Disabled'}"
    )

    # Check for database file
    db_path = settings.database.path
    db_exists = os.path.exists(db_path)
    db_size = os.path.getsize(db_path) if db_exists else 0
    db_size_mb = db_size / (1024 * 1024) if db_exists else 0

    console.print()
    console.print("[bold]Database Status:[/bold]")
    console.print(f"Database Exists: {'Yes' if db_exists else 'No'}")
    if db_exists:
        console.print(f"Database Size: {db_size_mb:.2f} MB")


@utility_app.command("cache")
def manage_cache(
    clear: bool = typer.Option(
        False, "--clear", "-c", help="Clear the embedding cache"
    ),
    info: bool = typer.Option(True, "--info", "-i", help="Show cache info"),
):
    """Manage the embedding cache."""

    # Get the embedding cache using type-based dependency injection
    from help_rag.infrastructure.embedding.embedding_cache import EmbeddingCache

    embedding_cache = container[EmbeddingCache]

    if clear:
        # Clear the cache
        async def clear_cache():
            count = await embedding_cache.clear_old_entries()
            return count

        cleared_count = asyncio.run(clear_cache())
        console.print(
            f"[bold green]Cleared {cleared_count} entries from embedding cache[/bold green]"
        )

    if info:
        # Get cache info
        async def get_cache_info():
            stats = await embedding_cache.get_stats()
            return stats

        stats = asyncio.run(get_cache_info())
        console.print("[bold]Embedding Cache Stats:[/bold]")
        console.print(f"Total entries: {stats.get('count', 0)}")
        for provider, count in stats.get("providers", {}).items():
            console.print(f"  {provider}: {count} entries")


@utility_app.command("test-db")
def test_db():
    """Test database connection and display table schema."""

    async def test_database():
        db_path = settings.database.path

        if not os.path.exists(db_path):
            console.print(f"[bold red]Database file not found: {db_path}[/bold red]")
            return

        import aiosqlite

        # Connect to the database
        async with aiosqlite.connect(db_path) as conn:
            # Get table list
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                tables = await cursor.fetchall()

            # Display table list
            console.print("[bold]Tables in database:[/bold]")
            for table in tables:
                console.print(f"- {table[0]}")

            # Get and display schema for each table
            console.print("\n[bold]Table schemas:[/bold]")
            for table in tables:
                table_name = table[0]
                async with conn.execute(f"PRAGMA table_info({table_name})") as cursor:
                    columns = await cursor.fetchall()

                console.print(f"\n[bold cyan]{table_name}[/bold cyan]")

                # Create a table to display the schema
                schema_table = Table()
                schema_table.add_column("CID")
                schema_table.add_column("Name")
                schema_table.add_column("Type")
                schema_table.add_column("Not Null")
                schema_table.add_column("Default")
                schema_table.add_column("Primary Key")

                for col in columns:
                    schema_table.add_row(
                        str(col[0]),
                        col[1],
                        col[2],
                        "Yes" if col[3] else "No",
                        str(col[4]) if col[4] is not None else "NULL",
                        "Yes" if col[5] else "No",
                    )

                console.print(schema_table)

    asyncio.run(test_database())


@utility_app.command("test-embed")
def test_embed(
    text: str = typer.Argument(..., help="Text to embed"),
):
    """Test the embedding service with a text input."""

    # Get the embedding service using type-based dependency injection
    from help_rag.domain.services.embedding_service import EmbeddingService

    # Type-safe access with proper typing
    embedding_service = container[EmbeddingService]

    # Create the embedding
    async def create_embedding():
        embedding = await embedding_service.embed_text(text)
        return embedding

    embedding = asyncio.run(create_embedding())

    # Display the embedding
    console.print(f"[bold]Input text:[/bold] {text}")
    console.print()
    console.print(f"[bold]Embedding dimension:[/bold] {len(embedding)}")
    console.print("[bold]Embedding preview:[/bold]")
    preview = ", ".join([f"{v:.4f}" for v in embedding[:10]])
    console.print(f"[{preview}, ...]")


def register_commands(app: typer.Typer):
    """Register utility commands with the main app."""
    app.add_typer(utility_app, name="utils", help="Utility commands")
