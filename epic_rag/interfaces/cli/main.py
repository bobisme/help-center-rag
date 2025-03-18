"""Command-line interface for Epic Documentation RAG system."""

import asyncio
import time
import os
import shutil
import datetime
import aiosqlite
from typing import Optional, List, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown
from rich.prompt import Confirm
from loguru import logger

from ...domain.models.document import Document
from ...infrastructure.config.settings import settings
from ...infrastructure.container import container, setup_container
from ...application.use_cases.ingest_document import IngestDocumentUseCase
from ...application.use_cases.retrieve_context import RetrieveContextUseCase

# Create Typer app with subcommands
app = typer.Typer(
    name="epic-rag",
    help="Epic Documentation RAG System",
    add_completion=False,
)

# Create database subcommand group
db_app = typer.Typer(help="Database maintenance commands")
app.add_typer(db_app, name="db", short_help="Database maintenance and utilities")

# Rich console for pretty output
console = Console()

# Configure loguru
logger.configure(
    handlers=[
        {
            "sink": lambda msg: console.print(
                f"[dim]{msg}[/dim]", markup=True, highlight=False
            )
        }
    ]
)


@app.callback()
def callback():
    """Epic Documentation RAG System.

    A retrieval-augmented generation system for Epic documentation
    using Anthropic's Contextual Retrieval methodology.
    """
    # Initialize the service container
    setup_container()


@app.command("ingest")
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
    import glob
    import os
    from pathlib import Path

    # Find all markdown files matching the pattern
    file_paths = glob.glob(os.path.join(source_dir, pattern))

    # Sort by modification time (newest first)
    file_paths = sorted(file_paths, key=os.path.getmtime, reverse=True)

    # Apply limit if specified
    if limit:
        file_paths = file_paths[:limit]

    # Show summary
    console.print(
        f"Found [bold]{len(file_paths)}[/bold] files in [cyan]{source_dir}[/cyan]"
    )

    # Get required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    chunking_service = container.get("chunking_service")
    embedding_service = container.get("embedding_service")

    # Create use case
    use_case = IngestDocumentUseCase(
        document_repository=document_repository,
        vector_repository=vector_repository,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )

    # Process all documents
    async def process_all_documents():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents", total=len(file_paths))

            results = []
            for file_path in file_paths:
                # Read the file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract title from the filename or first heading
                filename = os.path.basename(file_path)
                title = Path(filename).stem.replace("_", " ").title()

                # Create the document
                document = Document(
                    title=title,
                    content=content,
                    metadata={
                        "source_path": file_path,
                        "file_type": "markdown",
                        "filename": filename,
                    },
                )

                # Process the document
                progress.update(task, description=f"Processing {filename}")
                try:
                    result = await use_case.execute(
                        document=document,
                        dynamic_chunking=dynamic_chunking,
                        min_chunk_size=min_chunk_size,
                        max_chunk_size=max_chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    results.append(result)
                except Exception as e:
                    console.print(
                        f"[bold red]Error processing {filename}:[/bold red] {str(e)}"
                    )

                progress.update(task, advance=1)

            return results

    # Run the processing
    results = asyncio.run(process_all_documents())

    # Show results
    total_chunks = sum(doc.chunk_count for doc in results)

    console.print()
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Documents Processed", str(len(results)))
    table.add_row("Total Chunks Created", str(total_chunks))
    table.add_row(
        "Avg Chunks Per Document",
        f"{total_chunks / len(results):.2f}" if results else "0",
    )
    table.add_row("Chunking Method", "Dynamic" if dynamic_chunking else "Fixed")

    console.print(table)


@app.command("query")
def query_system(
    query_text: str = typer.Argument(..., help="The query to search for"),
    first_stage_k: int = typer.Option(
        20,
        "--first-stage-k",
        "-k1",
        help="Number of documents to retrieve in first stage",
    ),
    second_stage_k: int = typer.Option(
        5,
        "--second-stage-k",
        "-k2",
        help="Number of documents to retrieve in second stage",
    ),
    min_relevance_score: float = typer.Option(
        0.7, "--min-relevance", "-r", help="Minimum relevance score for filtering"
    ),
    use_query_transformation: bool = typer.Option(
        True,
        "--transform-query/--no-transform-query",
        help="Transform the query to better match document corpus",
    ),
    merge_related_chunks: bool = typer.Option(
        True,
        "--merge-chunks/--no-merge-chunks",
        help="Merge semantically related chunks",
    ),
    show_detailed_results: bool = typer.Option(
        False, "--show-details", "-d", help="Show detailed retrieval results"
    ),
):
    """Query the RAG system with a natural language question."""
    # Get required services
    embedding_service = container.get("embedding_service")
    retrieval_service = container.get("retrieval_service")

    # Create use case
    use_case = RetrieveContextUseCase(
        embedding_service=embedding_service, retrieval_service=retrieval_service
    )

    # Process the query
    async def process_query():
        with Progress(
            TextColumn("[bold blue]{task.description}"), BarColumn(), console=console
        ) as progress:
            task = progress.add_task("Processing query", total=100)

            # Update progress
            progress.update(task, advance=10, description="Embedding query")

            # Execute the query
            result = await use_case.execute(
                query_text=query_text,
                first_stage_k=first_stage_k,
                second_stage_k=second_stage_k,
                min_relevance_score=min_relevance_score,
                use_query_transformation=use_query_transformation,
                merge_related_chunks=merge_related_chunks,
            )

            progress.update(task, completed=100, description="Complete")
            return result

    # Run the query processing
    result = asyncio.run(process_query())

    # Show results
    console.print()

    if not result.final_chunks:
        console.print("[bold red]No results found.[/bold red]")
        return

    # Display the transformed query if available
    if result.query.transformed_text and use_query_transformation:
        console.print(
            Panel(
                f"[italic]{result.query.transformed_text}[/italic]",
                title="Transformed Query",
                border_style="blue",
            )
        )

    # Display merged content
    if result.merged_content:
        console.print(
            Panel(
                result.merged_content, title="Retrieved Context", border_style="green"
            )
        )
    else:
        # Display individual chunks if no merged content
        for i, chunk in enumerate(result.final_chunks, 1):
            console.print(
                Panel(
                    chunk.content,
                    title=f"Result {i} (Score: {chunk.relevance_score:.2f})",
                    border_style="green",
                )
            )

    # Show detailed metrics if requested
    if show_detailed_results:
        console.print()
        table = Table(title="Retrieval Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Latency", f"{result.total_latency_ms:.2f}ms")
        table.add_row(
            "Retrieval Latency",
            (
                f"{result.retrieval_latency_ms:.2f}ms"
                if result.retrieval_latency_ms
                else "N/A"
            ),
        )
        table.add_row(
            "Processing Latency",
            (
                f"{result.processing_latency_ms:.2f}ms"
                if result.processing_latency_ms
                else "N/A"
            ),
        )
        table.add_row(
            "First Stage Results", str(len(result.first_stage_results.chunks))
        )
        table.add_row(
            "Second Stage Results",
            str(
                len(result.second_stage_results.chunks)
                if result.second_stage_results
                else 0
            ),
        )
        table.add_row("Final Chunks", str(len(result.final_chunks)))

        console.print(table)


@app.command("chunks")
def visualize_chunks(
    file_path: str = typer.Option(
        ..., "--file", "-f", help="Path to markdown file to chunk"
    ),
    dynamic: bool = typer.Option(
        True, "--dynamic/--fixed", help="Use dynamic chunking or fixed-size chunking"
    ),
    min_chunk_size: int = typer.Option(
        300, "--min-size", help="Minimum chunk size (for dynamic chunking)"
    ),
    max_chunk_size: int = typer.Option(
        800,
        "--max-size",
        help="Maximum chunk size (for dynamic) or chunk size (for fixed)",
    ),
    chunk_overlap: int = typer.Option(
        50, "--overlap", help="Chunk overlap (for fixed-size chunking)"
    ),
    show_metadata: bool = typer.Option(
        False, "--metadata", "-m", help="Show chunk metadata"
    ),
):
    """Visualize chunks for a markdown file."""
    import os
    import asyncio
    import uuid
    from pathlib import Path
    from rich.markdown import Markdown

    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist.")
        return

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract title from filename or first heading
    filename = os.path.basename(file_path)
    title = Path(filename).stem.replace("_", " ").title()

    # Create a document
    from ...domain.models.document import Document

    document = Document(
        id=str(uuid.uuid4()),
        title=title,
        content=content,
        metadata={
            "source_path": file_path,
            "file_type": "markdown",
            "filename": filename,
        },
    )

    # Get chunking service
    chunking_service = container.get("chunking_service")

    # Process chunks based on chunking method
    async def process_chunks():
        if dynamic:
            return await chunking_service.dynamic_chunk_document(
                document=document,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
            )
        else:
            return await chunking_service.chunk_document(
                document=document,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )

    # Run the processing
    chunks = asyncio.run(process_chunks())

    # Display results
    console.print()
    console.print(f"[bold]File:[/bold] {file_path}")
    console.print(f"[bold]Title:[/bold] {title}")
    console.print(f"[bold]Chunking Method:[/bold] {'Dynamic' if dynamic else 'Fixed'}")
    console.print(f"[bold]Number of Chunks:[/bold] {len(chunks)}")
    console.print()

    # Display each chunk
    for i, chunk in enumerate(chunks, 1):
        # Create panel for chunk content
        panel_title = f"Chunk {i}/{len(chunks)}"
        if "title" in chunk.metadata:
            panel_title += f" - {chunk.metadata['title']}"

        # Show chunk content
        console.print(
            Panel(
                Markdown(chunk.content),
                title=panel_title,
                border_style="blue",
                expand=False,
            )
        )

        # Show metadata if requested
        if show_metadata:
            metadata_table = Table(title="Chunk Metadata")
            metadata_table.add_column("Key", style="cyan")
            metadata_table.add_column("Value", style="green")

            for key, value in chunk.metadata.items():
                # Skip verbose entries
                if key == "document_id" or key == "original_chunk_ids":
                    continue

                metadata_table.add_row(key, str(value))

            console.print(metadata_table)

        # Add separator between chunks
        if i < len(chunks):
            console.print()


@app.command("test-db")
def test_database():
    """Test the document storage by adding a test document."""
    import uuid
    from datetime import datetime

    # Get chunking service and document repository
    chunking_service = container.get("chunking_service")
    document_repository = container.get("document_repository")

    # Create a test document
    document_id = str(uuid.uuid4())
    test_content = """\
# Test Document

This is a test document created to verify database functionality.

## Section 1

This is the first section of the test document.

## Section 2

This is the second section with a list:

1. Item one
2. Item two
3. Item three

## Section 3

This is the final section of our test document.
"""

    # Create document

    document = Document(
        id=document_id,
        title="Test Document",
        content=test_content,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"test": True, "created_by": "CLI", "purpose": "Database testing"},
    )

    # Process the document
    async def process_document():
        # Save document
        console.print("[bold blue]Saving document to database...[/bold blue]")
        saved_doc = await document_repository.save_document(document)

        # Generate chunks
        console.print("[bold blue]Generating chunks...[/bold blue]")
        chunks = await chunking_service.dynamic_chunk_document(
            document=saved_doc, min_chunk_size=200, max_chunk_size=500
        )

        # Save chunks
        console.print(
            f"[bold blue]Saving {len(chunks)} chunks to database...[/bold blue]"
        )
        for chunk in chunks:
            await document_repository.save_chunk(chunk)

        # Get statistics
        stats = await document_repository.get_statistics()

        return {"document": saved_doc, "chunks": chunks, "stats": stats}

    # Run the processing
    results = asyncio.run(process_document())

    # Show results
    console.print()
    console.print(
        Panel(
            f"[bold green]Successfully saved test document and chunks[/bold green]\n\n"
            f"Document ID: {results['document'].id}\n"
            f"Title: {results['document'].title}\n"
            f"Chunks: {len(results['chunks'])}\n"
            f"Document Count: {results['stats']['document_count']['value']}\n"
            f"Chunk Count: {results['stats']['chunk_count']['value']}",
            title="Database Test Results",
            border_style="green",
        )
    )


@app.command("test-embed")
def test_embedding(
    text: str = typer.Argument(..., help="Text to embed"),
    compare_with: Optional[str] = typer.Option(
        None, "--compare", "-c", help="Optional text to compare similarity with"
    ),
    provider: str = typer.Option(
        "current",
        "--provider",
        "-p",
        help="Embedding provider to use (huggingface, openai, gemini, or current for using the configured provider)",
    ),
):
    """Test the embedding service by embedding text and optionally comparing with another text."""
    import numpy as np
    from epic_rag.infrastructure.config.settings import settings as global_settings

    # Backup original provider
    original_provider = global_settings.embedding.provider

    # Set provider if specified
    if provider != "current":
        if provider.lower() not in ["huggingface", "openai", "gemini"]:
            console.print(f"[bold red]Error:[/bold red] Invalid provider '{provider}'.")
            console.print("Supported providers: huggingface, openai, gemini")
            return

        global_settings.embedding.provider = provider.lower()
        setup_container()

    # Get embedding service
    try:
        embedding_service = container.get("embedding_service")
    except KeyError:
        console.print("[bold red]Error:[/bold red] Embedding service not registered.")
        console.print(f"Make sure {provider.upper()}_API_KEY is properly configured.")

        # Restore original provider
        if provider != "current":
            global_settings.embedding.provider = original_provider
            setup_container()
        return

    # Embed the text
    async def run_embedding():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating embedding", total=100)

            # Generate embedding for input text
            progress.update(task, advance=30, description="Embedding input text")
            embedding = await embedding_service.embed_text(text)

            progress.update(task, advance=40, description="Processing")

            # If comparison text is provided, generate embedding for it and calculate similarity
            comparison_embedding = None
            similarity = None

            if compare_with:
                progress.update(task, description="Embedding comparison text")
                comparison_embedding = await embedding_service.embed_text(compare_with)

                # Calculate similarity
                progress.update(task, description="Calculating similarity")
                similarity = await embedding_service.get_embedding_similarity(
                    embedding, comparison_embedding
                )

            progress.update(task, completed=100, description="Complete")

            return {
                "text": text,
                "embedding": embedding,
                "dimensions": len(embedding),
                "compare_with": compare_with,
                "comparison_embedding": comparison_embedding,
                "similarity": similarity,
            }

    # Run the embedding
    result = asyncio.run(run_embedding())

    # Show results
    console.print()

    # Get appropriate model name based on provider
    model_name = settings.embedding.model
    if settings.embedding.provider.lower() == "openai":
        model_name = settings.embedding.openai_model
    elif settings.embedding.provider.lower() == "gemini":
        model_name = settings.embedding.gemini_model

    console.print(
        Panel(
            f"[bold]Model:[/bold] {model_name}\n"
            f"[bold]Provider:[/bold] {settings.embedding.provider}\n"
            f"[bold]Dimensions:[/bold] {result['dimensions']}",
            title="Embedding Information",
            border_style="blue",
        )
    )

    # Display preview of the embedding vector
    console.print()
    console.print("[bold]Text:[/bold]")
    console.print(text)

    # Show embedding preview (first few dimensions)
    preview_count = min(5, len(result["embedding"]))
    preview = ", ".join(f"{result['embedding'][i]:.6f}" for i in range(preview_count))

    console.print()
    console.print(
        f"[bold]Embedding Vector[/bold] (first {preview_count} of {result['dimensions']} dimensions):"
    )
    console.print(f"[dim]{preview}...[/dim]")

    # Display vector statistics
    embedding_array = np.array(result["embedding"])
    stats = {
        "Mean": float(np.mean(embedding_array)),
        "Std Dev": float(np.std(embedding_array)),
        "Min": float(np.min(embedding_array)),
        "Max": float(np.max(embedding_array)),
        "L2 Norm": float(np.linalg.norm(embedding_array)),
    }

    console.print()
    table = Table(title="Vector Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        table.add_row(key, f"{value:.6f}")

    console.print(table)

    # Show similarity if comparison text was provided
    if result["similarity"] is not None:
        console.print()
        console.print("[bold]Comparison Text:[/bold]")
        console.print(compare_with)

        console.print()
        similarity_percentage = result["similarity"] * 100
        similarity_level = (
            "High"
            if similarity_percentage > 80
            else "Medium" if similarity_percentage > 50 else "Low"
        )

        console.print(
            Panel(
                f"[bold green]{similarity_percentage:.2f}%[/bold green]\n\n"
                f"Similarity Level: [bold]{similarity_level}[/bold]",
                title="Semantic Similarity",
                border_style=(
                    "green"
                    if similarity_percentage > 70
                    else "yellow" if similarity_percentage > 40 else "red"
                ),
            )
        )

    # Restore original provider if changed
    if provider != "current":
        from epic_rag.infrastructure.config.settings import settings as global_settings

        global_settings.embedding.provider = original_provider
        setup_container()


@app.command("cache")
def manage_cache(
    action: str = typer.Argument("stats", help="Action to perform: stats, clear"),
    days: int = typer.Option(
        30, "--days", "-d", help="Days to keep when clearing old entries"
    ),
):
    """Manage the embedding cache.

    Actions:
    - stats: Show cache statistics
    - clear: Clear old entries from the cache
    """
    if not settings.embedding.cache.enabled:
        console.print("[bold red]Error:[/bold red] Cache is disabled in settings.")
        return

    try:
        # Get embedding service
        embedding_service = container.get("embedding_service")

        # Check if it's a cached service
        if not hasattr(embedding_service, "get_cache_stats"):
            console.print(
                "[bold red]Error:[/bold red] Embedding service does not support caching."
            )
            return

        # Get cache from EmbeddingCache class
        from epic_rag.infrastructure.embedding.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(
            settings=settings,
            memory_cache_size=settings.embedding.cache.memory_size,
            cache_expiration_days=settings.embedding.cache.expiration_days,
        )

        # Perform action
        if action.lower() == "stats":
            asyncio.run(_show_cache_stats(embedding_service))
        elif action.lower() == "clear":
            asyncio.run(_clear_cache_entries(cache, days))
        else:
            console.print(f"[bold red]Error:[/bold red] Unknown action: {action}")
            console.print("Available actions: stats, clear")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


async def _show_cache_stats(embedding_service):
    """Show statistics about the embedding cache."""
    # Get cache stats
    stats = await embedding_service.get_cache_stats()

    if not stats:
        console.print("[bold yellow]No cache statistics available.[/bold yellow]")
        return

    # Show general information
    console.print()
    console.print(
        Panel(
            f"[bold]Cache Status:[/bold] Enabled\n"
            f"[bold]Memory Cache Size:[/bold] {stats['memory_max_size']} entries\n"
            f"[bold]Memory Cache Usage:[/bold] {stats['memory_entries']} entries "
            f"({stats['memory_entries'] / stats['memory_max_size'] * 100:.1f}%)\n"
            f"[bold]Total Entries:[/bold] {stats['total_entries']}\n"
            f"[bold]Storage Size:[/bold] {stats['db_size_bytes'] / (1024*1024):.2f} MB\n"
            f"[bold]Oldest Entry:[/bold] {stats['oldest_entry'] or 'None'}\n"
            f"[bold]Newest Entry:[/bold] {stats['newest_entry'] or 'None'}",
            title="Embedding Cache Statistics",
            border_style="blue",
        )
    )

    # Show provider breakdown
    if stats.get("by_provider"):
        console.print()
        table = Table(title="Cache Entries by Provider")
        table.add_column("Provider", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        total = stats["total_entries"] or 1  # Avoid division by zero
        for provider, count in stats["by_provider"].items():
            table.add_row(
                provider,
                str(count),
                f"{count / total * 100:.1f}%",
            )

        console.print(table)

    # Show model breakdown
    if stats.get("by_model"):
        console.print()
        table = Table(title="Cache Entries by Model")
        table.add_column("Model", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        total = stats["total_entries"] or 1  # Avoid division by zero
        for model, count in stats["by_model"].items():
            table.add_row(
                model,
                str(count),
                f"{count / total * 100:.1f}%",
            )

        console.print(table)


async def _clear_cache_entries(cache, days):
    """Clear old entries from the cache."""
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Clearing old cache entries...", total=100)

        # Update cache expiration days
        cache._cache_expiration_days = days

        # Clear old entries
        progress.update(task, advance=50)
        count = await cache.clear_old_entries()

        progress.update(task, completed=100)

    console.print(
        f"[bold green]Cleared {count} old entries older than {days} days.[/bold green]"
    )


@app.command("transform-query")
def transform_query(
    query_text: str = typer.Argument(..., help="Query text to transform"),
    model: str = typer.Option(
        "gemma3:27b", "--model", "-m", help="Ollama model to use for transformation"
    ),
):
    """Test query transformation functionality using the LLM."""
    from epic_rag.domain.models.retrieval import Query

    # Update the LLM model setting
    settings.llm.model = model

    # Setup container with the specified model
    setup_container()

    # Get LLM service
    llm_service = container.get("llm_service")

    # Transform query
    async def run_transform():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Transforming query", total=100)

            # Update progress
            progress.update(task, advance=10, description="Processing with LLM")

            # Transform query
            transformed_text = await llm_service.transform_query(query_text)

            progress.update(task, completed=100, description="Complete")
            return transformed_text

    # Run transformation
    try:
        transformed_text = asyncio.run(run_transform())

        # Display results
        console.print()
        console.print(
            Panel(
                f"[bold]Original Query:[/bold]\n{query_text}\n\n"
                f"[bold]Transformed Query:[/bold]\n{transformed_text}",
                title=f"Query Transformation ({model})",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("Make sure Ollama is running with the specified model.")


@app.command("bm25")
def test_bm25(
    query_text: str = typer.Argument(..., help="Query text to search"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results to return"
    ),
    show_full_content: bool = typer.Option(
        False, "--full-content", "-f", help="Show full chunk content instead of preview"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Show detailed debug information"
    ),
):
    """Test BM25 search functionality."""
    from epic_rag.domain.models.retrieval import Query
    import logging

    # Set up debug logging if requested
    if debug:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Get BM25 service
    bm25_service = container.get("bm25_search_service")

    # Create query
    query = Query(
        text=query_text,
        metadata={"source": "cli"},
    )

    # Run search
    async def run_search():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("BM25 search", total=100)

            # First, ensure all documents are indexed
            progress.update(task, advance=30, description="Indexing documents")
            await bm25_service.reindex_all()

            # Show corpus debug info
            if debug:
                corpus_content = getattr(bm25_service, "corpus", [])
                console.print(
                    f"[bold cyan]BM25 Corpus Size:[/bold cyan] {len(corpus_content)}"
                )
                if len(corpus_content) > 0:
                    console.print("[bold cyan]First document sample:[/bold cyan]")
                    console.print(corpus_content[0][:200] + "...")

            # Run BM25 search
            progress.update(task, advance=30, description="Searching")
            result = await bm25_service.search(query, limit=limit)

            progress.update(task, completed=100, description="Complete")
            return result

    # Run search
    try:
        result = asyncio.run(run_search())

        # Direct BM25S test if debugging is enabled
        if debug:
            console.print("\n[bold blue]Direct BM25S Test:[/bold blue]")

            # Import BM25S directly
            import bm25s

            # Get the corpus directly from the service
            corpus = getattr(bm25_service, "corpus", [])

            if not corpus:
                console.print("[yellow]Corpus is empty, can't run direct test[/yellow]")
            else:
                console.print(f"Corpus size: {len(corpus)}")
                console.print("Building direct BM25 model...")

                # Create a fresh BM25 model
                model = bm25s.BM25(corpus=corpus)

                # Tokenize corpus and query
                console.print("Tokenizing corpus and query...")
                tokenized_corpus = bm25s.tokenize(corpus)
                tokenized_query = bm25s.tokenize([query.text])

                # Index and retrieve
                console.print("Indexing corpus...")
                model.index(tokenized_corpus)

                console.print(f"Running search for: '{query.text}'")
                k = min(3, len(corpus))
                doc_indices, scores = model.retrieve(tokenized_query, k=k)

                console.print(f"Got results: {doc_indices}")
                console.print(f"Scores: {scores}")

                # Show first document match if any
                if doc_indices and doc_indices[0]:
                    idx = doc_indices[0][0]
                    score = scores[0][0]
                    console.print(f"Top match (idx={idx}, score={score}):")
                    console.print(corpus[idx][:200] + "...")

        # Display results
        console.print()
        console.print(
            Panel(
                f"Found [bold]{len(result.chunks)}[/bold] matches",
                title=f"BM25 Search Results for: {query_text}",
                border_style="blue",
            )
        )

        # Show all results
        if len(result.chunks) == 0:
            console.print("[yellow]No results found.[/yellow]")
        else:
            for i, chunk in enumerate(result.chunks, 1):
                # Prepare the content (full or preview)
                if show_full_content:
                    content = chunk.content
                else:
                    # Extract a preview (first 200 chars)
                    content = chunk.content[:200] + (
                        "..." if len(chunk.content) > 200 else ""
                    )

                # Show the result
                console.print(
                    Panel(
                        f"[bold]Score:[/bold] {chunk.relevance_score:.4f}\n"
                        f"[bold]Document:[/bold] {chunk.metadata.get('title', 'Untitled')}\n"
                        f"[bold]Content:[/bold]\n{content}",
                        title=f"Result {i}/{len(result.chunks)}",
                        border_style="green" if i == 1 else "blue",
                    )
                )

        # Show timing information
        console.print(f"Search completed in {result.latency_ms:.2f}ms")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command("hybrid-search")
def test_hybrid_search(
    query_text: str = typer.Argument(..., help="Query text to search"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results to return"
    ),
    bm25_weight: float = typer.Option(
        0.4, "--bm25-weight", help="Weight for BM25 results (0.0-1.0)"
    ),
    vector_weight: float = typer.Option(
        0.6, "--vector-weight", help="Weight for vector results (0.0-1.0)"
    ),
    show_separate_results: bool = typer.Option(
        False, "--show-separate", "-s", help="Show vector and BM25 results separately"
    ),
    show_full_content: bool = typer.Option(
        False, "--full-content", "-f", help="Show full chunk content instead of preview"
    ),
    rerank: bool = typer.Option(
        False, "--rerank", "-r", help="Apply reranking to results"
    ),
):
    """Test hybrid search with BM25 and vector search with rank fusion."""
    from epic_rag.domain.models.retrieval import Query, RetrievalResult

    # Get required services
    bm25_service = container.get("bm25_search_service")
    embedding_service = container.get("embedding_service")
    vector_repository = container.get("vector_repository")
    document_repository = container.get("document_repository")
    rank_fusion_service = container.get("rank_fusion_service")

    # Get reranker service if needed
    reranker_service = None
    if rerank:
        try:
            # Force-enable reranker for this test
            settings.retrieval.reranker.enabled = True
            if not container.has("reranker_service"):
                from epic_rag.infrastructure.reranker.cross_encoder_reranker_service import (
                    CrossEncoderRerankerService,
                )

                container.register(
                    "reranker_service",
                    CrossEncoderRerankerService(
                        model_name=settings.retrieval.reranker.model_name
                    ),
                )
            reranker_service = container.get("reranker_service")
        except Exception as e:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Could not load reranker: {e}"
            )

    # Create query
    query = Query(
        text=query_text,
        metadata={"source": "cli"},
    )

    # Run search
    async def run_search():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Hybrid search", total=100)

            # Ensure all documents are indexed in BM25
            progress.update(task, advance=20, description="Indexing documents")
            await bm25_service.reindex_all()

            # Embed the query for vector search
            progress.update(task, advance=20, description="Embedding query")
            query_with_embedding = await embedding_service.embed_query(query)

            # Run BM25 search
            progress.update(task, advance=20, description="BM25 search")
            bm25_results = await bm25_service.search(query, limit=limit)

            # Run vector search
            progress.update(task, advance=20, description="Vector search")
            vector_chunks = await vector_repository.search_similar(
                query=query_with_embedding, limit=limit
            )

            # Create vector results
            for chunk in vector_chunks:
                # Get full chunk content from document repository
                full_chunk = await document_repository.get_chunk(chunk.id)
                if full_chunk:
                    # Update the chunk content while keeping the relevance score
                    chunk.content = full_chunk.content
                    # Copy any missing metadata
                    for key, value in full_chunk.metadata.items():
                        if key not in chunk.metadata:
                            chunk.metadata[key] = value

            vector_results = RetrievalResult(
                query_id=query.id,
                chunks=vector_chunks,
                latency_ms=0.0,  # Initialize with 0
            )

            # Fuse results
            progress.update(task, advance=15, description="Fusing results")
            fused_results = await rank_fusion_service.fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
            )

            # Apply reranking if enabled
            reranked_results = None
            if rerank and reranker_service:
                progress.update(task, advance=5, description="Reranking results")
                reranked_chunks = await reranker_service.rerank(
                    query=query,
                    chunks=fused_results.chunks,
                    top_k=settings.retrieval.reranker.top_k,
                )

                # Create new result with reranked chunks
                reranked_results = RetrievalResult(
                    query_id=query.id,
                    chunks=reranked_chunks,
                    latency_ms=fused_results.latency_ms,  # We'll add reranking time later
                )

            progress.update(task, completed=100, description="Complete")
            return {
                "vector": vector_results,
                "bm25": bm25_results,
                "fused": fused_results,
                "reranked": reranked_results,
            }

    # Run search
    try:
        results = asyncio.run(run_search())

        # Display results
        console.print()

        # Base result panel content
        panel_content = (
            f"[bold]Query:[/bold] {query_text}\n"
            f"[bold]BM25 Results:[/bold] {len(results['bm25'].chunks)}\n"
            f"[bold]Vector Results:[/bold] {len(results['vector'].chunks)}\n"
            f"[bold]Fused Results:[/bold] {len(results['fused'].chunks)}\n"
        )

        # Add reranking info if present
        if rerank and results["reranked"]:
            panel_content += (
                f"[bold]Reranked Results:[/bold] {len(results['reranked'].chunks)}\n"
            )
            panel_content += f"[bold]Reranker Model:[/bold] {settings.retrieval.reranker.model_name}\n"

        # Add weights
        panel_content += f"[bold]BM25 Weight:[/bold] {bm25_weight}\n"
        panel_content += f"[bold]Vector Weight:[/bold] {vector_weight}"

        console.print(
            Panel(
                panel_content,
                title="Hybrid Search Results",
                border_style="blue",
            )
        )

        # Show separate results if requested
        if show_separate_results:
            # Show BM25 results
            console.print()
            console.print("[bold blue]BM25 Search Results:[/bold blue]")
            for i, chunk in enumerate(results["bm25"].chunks[:5], 1):
                # Prepare content
                if show_full_content:
                    content = chunk.content
                else:
                    content = chunk.content[:200] + (
                        "..." if len(chunk.content) > 200 else ""
                    )

                console.print(
                    Panel(
                        f"[bold]Score:[/bold] {chunk.relevance_score:.4f}\n"
                        f"[bold]Content Preview:[/bold]\n{content}",
                        title=f"BM25 Result {i}",
                        border_style="yellow",
                    )
                )

            # Show vector results
            console.print()
            console.print("[bold blue]Vector Search Results:[/bold blue]")
            for i, chunk in enumerate(results["vector"].chunks[:5], 1):
                # Prepare content
                if show_full_content:
                    content = chunk.content
                else:
                    content = chunk.content[:200] + (
                        "..." if len(chunk.content) > 200 else ""
                    )

                console.print(
                    Panel(
                        f"[bold]Score:[/bold] {chunk.relevance_score:.4f}\n"
                        f"[bold]Content Preview:[/bold]\n{content}",
                        title=f"Vector Result {i}",
                        border_style="cyan",
                    )
                )

        # Determine which results to show (fused or reranked)
        final_results = (
            results["reranked"] if rerank and results["reranked"] else results["fused"]
        )
        result_type = "Reranked" if rerank and results["reranked"] else "Fused"

        # Show results
        console.print()
        console.print(f"[bold blue]{result_type} Results:[/bold blue]")
        for i, chunk in enumerate(final_results.chunks, 1):
            # Prepare the content (full or preview)
            if show_full_content:
                content = chunk.content
            else:
                content = chunk.content[:200] + (
                    "..." if len(chunk.content) > 200 else ""
                )

            # Get original rankings
            bm25_rank = "N/A"
            for j, bm25_chunk in enumerate(results["bm25"].chunks, 1):
                if bm25_chunk.id == chunk.id:
                    bm25_rank = str(j)
                    break

            vector_rank = "N/A"
            for j, vector_chunk in enumerate(results["vector"].chunks, 1):
                if vector_chunk.id == chunk.id:
                    vector_rank = str(j)
                    break

            # Get fusion rank if showing reranked results
            fusion_rank = "N/A"
            if rerank and results["reranked"]:
                for j, fused_chunk in enumerate(results["fused"].chunks, 1):
                    if fused_chunk.id == chunk.id:
                        fusion_rank = str(j)
                        break

            # Build panel content
            panel_content = (
                f"[bold]{result_type} Score:[/bold] {chunk.relevance_score:.4f}\n"
            )
            panel_content += f"[bold]BM25 Rank:[/bold] {bm25_rank}\n"
            panel_content += f"[bold]Vector Rank:[/bold] {vector_rank}\n"

            # Add fusion rank if showing reranked results
            if rerank and results["reranked"]:
                panel_content += f"[bold]Fusion Rank:[/bold] {fusion_rank}\n"

            panel_content += (
                f"[bold]Document:[/bold] {chunk.metadata.get('title', 'Untitled')}\n"
            )
            panel_content += f"[bold]Content:[/bold]\n{content}"

            console.print(
                Panel(
                    panel_content,
                    title=f"Result {i}/{len(final_results.chunks)}",
                    border_style="green" if i <= 3 else "blue",
                )
            )

        # Show timing information
        console.print()
        if results["bm25"].latency_ms is not None:
            console.print(f"BM25 latency: {results['bm25'].latency_ms:.2f}ms")
        else:
            console.print("BM25 latency: N/A")

        if results["vector"].latency_ms is not None:
            console.print(f"Vector latency: {results['vector'].latency_ms:.2f}ms")
        else:
            console.print("Vector latency: N/A")

        if results["fused"].latency_ms is not None:
            console.print(f"Total latency: {results['fused'].latency_ms:.2f}ms")
        else:
            console.print("Total latency: N/A")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback

        console.print(traceback.format_exc())


@app.command("benchmark-bm25")
def benchmark_bm25(
    query_text: str = typer.Argument(..., help="Query text to search"),
    iterations: int = typer.Option(
        10, "--iterations", "-n", help="Number of iterations for benchmark"
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results to return"
    ),
    document_count: int = typer.Option(
        50, "--documents", "-d", help="Number of test documents to generate"
    ),
):
    """Benchmark different BM25 implementations."""
    import time
    import numpy as np
    import random
    import string
    import uuid
    from datetime import datetime

    from epic_rag.domain.models.retrieval import Query
    from epic_rag.domain.models.document import Document, DocumentChunk
    from epic_rag.infrastructure.search.bm25_search_service import BM25SearchService
    from epic_rag.infrastructure.search.bm25s_search_service import BM25SSearchService

    # Get document repository
    document_repository = container.get("document_repository")

    # Create both implementations
    bm25_service = BM25SearchService(document_repository=document_repository)
    bm25s_service = BM25SSearchService(document_repository=document_repository)

    # Create query
    query = Query(
        id=str(uuid.uuid4()),
        text=query_text,
        metadata={"source": "cli"},
    )

    # Run benchmark
    async def run_benchmark():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # First, clear existing documents and create test data
            prep_task = progress.add_task(
                "Preparing test data...", total=document_count
            )

            # Generate test data
            test_chunks = []
            topics = ["healthcare", "technology", "finance", "education", "sports"]

            # Keywords that will be in the query
            query_keywords = query_text.lower().split()

            for i in range(document_count):
                # Generate random content with some query keywords
                topic = random.choice(topics)
                words = [
                    random.choice(string.ascii_lowercase)
                    for _ in range(random.randint(100, 500))
                ]

                # Insert some query keywords randomly
                for keyword in query_keywords:
                    for _ in range(random.randint(0, 3)):
                        position = random.randint(0, len(words) - 1)
                        words[position] = keyword

                content = f"# Document about {topic}\n\nThis is a test document about {topic}. "
                content += " ".join(words)

                # Create a document chunk
                chunk = DocumentChunk(
                    id=f"test-{i}",
                    document_id=f"doc-{i}",
                    content=content,
                    metadata={"title": f"Test Document {i}", "topic": topic},
                    chunk_index=0,
                )

                test_chunks.append(chunk)
                progress.update(prep_task, advance=1)

            # Index documents
            progress.update(
                prep_task, completed=True, description="Indexing documents in BM25..."
            )
            await bm25_service.index_documents(test_chunks)

            progress.update(prep_task, description="Indexing documents in BM25S...")
            await bm25s_service.index_documents(test_chunks)

            progress.update(prep_task, description="Test data prepared")

            # Run benchmark iterations
            bm25_task = progress.add_task(
                "Benchmarking BM25 (rank-bm25)...", total=iterations
            )
            bm25s_task = progress.add_task(
                "Benchmarking BM25S (huggingface)...", total=iterations
            )

            # Warm-up run
            await bm25_service.search(query, limit=limit)
            await bm25s_service.search(query, limit=limit)

            # BM25 benchmark
            bm25_latencies = []
            for i in range(iterations):
                start_time = time.time()
                result = await bm25_service.search(query, limit=limit)
                latency = (time.time() - start_time) * 1000
                bm25_latencies.append(latency)
                progress.update(bm25_task, advance=1)

            # BM25S benchmark
            bm25s_latencies = []
            for i in range(iterations):
                start_time = time.time()
                result = await bm25s_service.search(query, limit=limit)
                latency = (time.time() - start_time) * 1000
                bm25s_latencies.append(latency)
                progress.update(bm25s_task, advance=1)

            # Run comparison search to verify result quality
            bm25_results = await bm25_service.search(query, limit=limit)
            bm25s_results = await bm25s_service.search(query, limit=limit)

            return {
                "bm25": {
                    "latencies": bm25_latencies,
                    "results": bm25_results,
                },
                "bm25s": {
                    "latencies": bm25s_latencies,
                    "results": bm25s_results,
                },
            }

    # Run benchmark
    try:
        results = asyncio.run(run_benchmark())

        # Calculate statistics
        bm25_latencies = results["bm25"]["latencies"]
        bm25s_latencies = results["bm25s"]["latencies"]

        bm25_avg = np.mean(bm25_latencies)
        bm25_min = np.min(bm25_latencies)
        bm25_max = np.max(bm25_latencies)
        bm25_stddev = np.std(bm25_latencies)

        bm25s_avg = np.mean(bm25s_latencies)
        bm25s_min = np.min(bm25s_latencies)
        bm25s_max = np.max(bm25s_latencies)
        bm25s_stddev = np.std(bm25s_latencies)

        speedup = bm25_avg / bm25s_avg if bm25s_avg > 0 else 0

        # Display results
        console.print()
        console.print(
            Panel(
                f"[bold]Query:[/bold] {query_text}\n"
                f"[bold]Iterations:[/bold] {iterations}\n"
                f"[bold]Results Limit:[/bold] {limit}\n"
                f"[bold]Test Documents:[/bold] {document_count}",
                title="BM25 Implementations Benchmark",
                border_style="blue",
            )
        )

        # Show performance comparison
        console.print()
        table = Table(title="Performance Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("BM25 (rank-bm25)", style="green")
        table.add_column("BM25S (huggingface)", style="green")
        table.add_column("Speedup", style="yellow")

        table.add_row(
            "Average Latency",
            f"{bm25_avg:.2f} ms",
            f"{bm25s_avg:.2f} ms",
            f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower",
        )
        table.add_row("Min Latency", f"{bm25_min:.2f} ms", f"{bm25s_min:.2f} ms", "")
        table.add_row("Max Latency", f"{bm25_max:.2f} ms", f"{bm25s_max:.2f} ms", "")
        table.add_row(
            "Std Deviation", f"{bm25_stddev:.2f} ms", f"{bm25s_stddev:.2f} ms", ""
        )
        table.add_row(
            "Result Count",
            str(len(results["bm25"]["results"].chunks)),
            str(len(results["bm25s"]["results"].chunks)),
            "",
        )

        console.print(table)

        # Show result quality comparison (first 3 results)
        console.print()
        console.print("[bold blue]Top Results Comparison:[/bold blue]")

        # Create a side-by-side view of the first 3 results
        max_results = min(
            3,
            max(
                len(results["bm25"]["results"].chunks),
                len(results["bm25s"]["results"].chunks),
            ),
        )
        if max_results > 0:
            for i in range(max_results):
                bm25_chunk = (
                    results["bm25"]["results"].chunks[i]
                    if i < len(results["bm25"]["results"].chunks)
                    else None
                )
                bm25s_chunk = (
                    results["bm25s"]["results"].chunks[i]
                    if i < len(results["bm25s"]["results"].chunks)
                    else None
                )

                bm25_content = bm25_chunk.content[:200] + "..." if bm25_chunk else "N/A"
                bm25s_content = (
                    bm25s_chunk.content[:200] + "..." if bm25s_chunk else "N/A"
                )

                bm25_score = (
                    f"{bm25_chunk.relevance_score:.4f}" if bm25_chunk else "N/A"
                )
                bm25s_score = (
                    f"{bm25s_chunk.relevance_score:.4f}" if bm25s_chunk else "N/A"
                )

                console.print(f"[bold]Result {i+1}:[/bold]")
                console.print(f"[bold]BM25 Score:[/bold] {bm25_score}")
                console.print(f"[bold]BM25S Score:[/bold] {bm25s_score}")

                columns = Table.grid(padding=1)
                columns.add_column("BM25", style="green")
                columns.add_column("BM25S", style="blue")

                columns.add_row(bm25_content, bm25s_content)
                console.print(columns)
                console.print()
        else:
            console.print("[yellow]No results found to compare.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback

        console.print(traceback.format_exc())


@app.command("test-rerank")
def test_reranker(
    query_text: str = typer.Argument(..., help="Query text to search"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of initial results to retrieve"
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of results to keep after reranking"
    ),
    model: str = typer.Option(
        "mixedbread-ai/mxbai-rerank-large-v1",
        "--model",
        "-m",
        help="Cross-encoder model to use",
    ),
    show_full_content: bool = typer.Option(
        False, "--full-content", "-f", help="Show full chunk content instead of preview"
    ),
):
    """Test the reranker service on vector search results."""
    from epic_rag.domain.models.retrieval import Query, RetrievalResult
    from epic_rag.infrastructure.reranker.cross_encoder_reranker_service import (
        CrossEncoderRerankerService,
    )

    # Get required services
    embedding_service = container.get("embedding_service")
    vector_repository = container.get("vector_repository")
    document_repository = container.get("document_repository")

    # Create reranker with specified model
    try:
        reranker_service = CrossEncoderRerankerService(model_name=model)
    except Exception as e:
        console.print(f"[bold red]Error creating reranker:[/bold red] {str(e)}")
        console.print("Make sure the specified model is available.")
        return

    # Create query
    query = Query(
        text=query_text,
        metadata={"source": "cli"},
    )

    # Run test
    async def run_test():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Testing reranker", total=100)

            # Embed the query for vector search
            progress.update(task, advance=20, description="Embedding query")
            query_with_embedding = await embedding_service.embed_query(query)

            # Run vector search
            progress.update(task, advance=20, description="Vector search")
            vector_chunks = await vector_repository.search_similar(
                query=query_with_embedding, limit=limit
            )

            # Create vector results
            for chunk in vector_chunks:
                # Get full chunk content from document repository
                full_chunk = await document_repository.get_chunk(chunk.id)
                if full_chunk:
                    # Update the chunk content while keeping the relevance score
                    chunk.content = full_chunk.content
                    # Copy any missing metadata
                    for key, value in full_chunk.metadata.items():
                        if key not in chunk.metadata:
                            chunk.metadata[key] = value

            # Create vector result
            vector_results = RetrievalResult(
                query_id=query.id,
                chunks=vector_chunks,
                latency_ms=0.0,
            )

            # Apply reranking
            progress.update(task, advance=40, description="Reranking results")
            start_time = time.time()
            reranked_chunks = await reranker_service.rerank(
                query=query, chunks=vector_chunks, top_k=top_k
            )
            rerank_time = (time.time() - start_time) * 1000

            # Create reranked result
            reranked_results = RetrievalResult(
                query_id=query.id,
                chunks=reranked_chunks,
                latency_ms=rerank_time,
            )

            progress.update(task, completed=100, description="Complete")
            return {
                "vector": vector_results,
                "reranked": reranked_results,
            }

    # Run test
    try:
        results = asyncio.run(run_test())

        # Display results
        console.print()
        console.print(
            Panel(
                f"[bold]Query:[/bold] {query_text}\n"
                f"[bold]Model:[/bold] {model}\n"
                f"[bold]Initial Results:[/bold] {len(results['vector'].chunks)}\n"
                f"[bold]Reranked Results:[/bold] {len(results['reranked'].chunks)}\n"
                f"[bold]Reranking Time:[/bold] {results['reranked'].latency_ms:.2f}ms",
                title="Reranker Test Results",
                border_style="blue",
            )
        )

        # Show comparison
        console.print()
        console.print("[bold blue]Vector Search vs. Reranked Results:[/bold blue]")

        # Show side-by-side comparison of up to 5 results
        comparison_count = min(
            5, len(results["vector"].chunks), len(results["reranked"].chunks)
        )

        for i in range(comparison_count):
            vector_chunk = (
                results["vector"].chunks[i]
                if i < len(results["vector"].chunks)
                else None
            )
            reranked_chunk = (
                results["reranked"].chunks[i]
                if i < len(results["reranked"].chunks)
                else None
            )

            # Prepare content previews
            if show_full_content:
                vector_content = vector_chunk.content if vector_chunk else "N/A"
                reranked_content = reranked_chunk.content if reranked_chunk else "N/A"
            else:
                vector_content = (
                    vector_chunk.content[:200]
                    + ("..." if len(vector_chunk.content) > 200 else "")
                    if vector_chunk
                    else "N/A"
                )
                reranked_content = (
                    reranked_chunk.content[:200]
                    + ("..." if len(reranked_chunk.content) > 200 else "")
                    if reranked_chunk
                    else "N/A"
                )

            # Get scores
            vector_score = (
                f"{vector_chunk.relevance_score:.4f}" if vector_chunk else "N/A"
            )
            reranked_score = (
                f"{reranked_chunk.relevance_score:.4f}" if reranked_chunk else "N/A"
            )

            # Find reranked position in original vector results
            original_rank = "N/A"
            if reranked_chunk:
                for j, vec_chunk in enumerate(results["vector"].chunks, 1):
                    if vec_chunk.id == reranked_chunk.id:
                        original_rank = str(j)
                        break

            # Show comparison
            console.print(f"[bold]Result {i+1}:[/bold]")

            # Show side-by-side scores
            table = Table.grid(padding=1)
            table.add_column("Vector", style="cyan", width=30)
            table.add_column("Reranked", style="green", width=30)

            table.add_row(f"Score: {vector_score}", f"Score: {reranked_score}")
            if reranked_chunk:
                table.add_row("", f"Original Rank: {original_rank}")

            console.print(table)

            # Show side-by-side content
            columns = Table.grid(padding=1)
            columns.add_column("Vector Content", style="cyan")
            columns.add_column("Reranked Content", style="green")

            columns.add_row(vector_content, reranked_content)
            console.print(columns)
            console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback

        console.print(traceback.format_exc())


@app.command("zenml-run")
def run_zenml_pipeline(
    source_dir: Optional[str] = typer.Option(
        None, "--source-dir", "-s", help="Directory containing markdown files"
    ),
    pattern: str = typer.Option(
        "*.md", "--pattern", "-p", help="Pattern to match markdown files"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of files to process"
    ),
    dynamic_chunking: bool = typer.Option(
        True, "--dynamic-chunking/--fixed-chunking", help="Use dynamic chunking"
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
    apply_enrichment: bool = typer.Option(
        True,
        "--apply-enrichment/--skip-enrichment",
        help="Apply LLM-based contextual enrichment to chunks",
    ),
    query_file: Optional[str] = typer.Option(
        None, "--query-file", "-q", help="Optional file containing test queries"
    ),
    pipeline_name: str = typer.Option(
        "orchestration",
        "--pipeline",
        "-p",
        help="Pipeline to run (orchestration, document_processing, query_evaluation, help_center)",
    ),
):
    """Run the ZenML pipeline for the Epic Documentation RAG system."""
    import sys
    from ...infrastructure.zenml.components import register_custom_components

    # Register custom ZenML components
    register_custom_components()

    # Show pipeline info
    console.print(
        Panel(
            f"[bold]Running ZenML Pipeline: {pipeline_name}[/bold]\n\n"
            f"[cyan]Source Directory:[/cyan] {source_dir}\n"
            f"[cyan]File Pattern:[/cyan] {pattern}\n"
            f"[cyan]Chunking:[/cyan] {'Dynamic' if dynamic_chunking else 'Fixed'} "
            f"(min={min_chunk_size}, max={max_chunk_size}, overlap={chunk_overlap})\n"
            f"[cyan]Contextual Enrichment:[/cyan] {'Enabled' if apply_enrichment else 'Disabled'}\n"
            f"[cyan]Query File:[/cyan] {query_file or 'Auto-generated queries'}\n",
            title="ZenML Pipeline Execution",
            border_style="blue",
        )
    )

    try:
        # Choose which pipeline to run
        if pipeline_name.lower() == "orchestration":
            from ...application.pipelines.orchestration_pipeline import (
                orchestration_pipeline,
            )

            # Check if source_dir is provided
            if source_dir is None:
                console.print(
                    "[bold red]Error:[/bold red] Source directory is required for orchestration pipeline."
                )
                return

            # Run the pipeline
            console.print("[bold blue]Running orchestration pipeline...[/bold blue]")
            orchestration_pipeline(
                source_dir=source_dir,
                file_pattern=pattern,
                limit=limit,
                dynamic_chunking=dynamic_chunking,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                apply_enrichment=apply_enrichment,
                query_file=query_file,
            )
        elif pipeline_name.lower() == "document_processing":
            from ...application.pipelines.document_processing_pipeline import (
                document_processing_pipeline,
            )

            # Check if source_dir is provided
            if source_dir is None:
                console.print(
                    "[bold red]Error:[/bold red] Source directory is required for document processing pipeline."
                )
                return

            # Run the pipeline
            console.print(
                "[bold blue]Running document processing pipeline...[/bold blue]"
            )
            document_processing_pipeline(
                source_dir=source_dir,
                file_pattern=pattern,
                limit=limit,
                dynamic_chunking=dynamic_chunking,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                apply_enrichment=apply_enrichment,
            )
        elif pipeline_name.lower() == "query_evaluation":
            from ...application.pipelines.query_evaluation_pipeline import (
                query_evaluation_pipeline,
            )

            # Check if query file is provided
            if not query_file:
                console.print(
                    "[bold red]Error:[/bold red] Query file is required for query evaluation pipeline."
                )
                return

            # Run the pipeline
            console.print("[bold blue]Running query evaluation pipeline...[/bold blue]")
            query_evaluation_pipeline(
                query_file=query_file,
            )
        elif pipeline_name.lower() == "help_center":
            from ...application.pipelines.help_center_pipeline import (
                help_center_processing_pipeline,
            )

            # Default values for help center specific options
            json_path = typer.prompt(
                "Path to input JSON file", default="output/epic-docs.json"
            )
            output_dir = typer.prompt(
                "Output directory for markdown files", default="data/help_center"
            )
            images_dir = typer.prompt(
                "Directory containing images", default="output/images"
            )
            start_index = typer.prompt("Start index", default=0, type=int)

            # Run the pipeline
            console.print(
                "[bold blue]Running help center processing pipeline...[/bold blue]"
            )
            help_center_processing_pipeline(
                json_path=json_path,
                output_dir=output_dir,
                images_dir=images_dir,
                start_index=start_index,
                limit=limit,
                dynamic_chunking=dynamic_chunking,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                apply_enrichment=apply_enrichment,
            )
        else:
            console.print(
                f"[bold red]Error:[/bold red] Unknown pipeline: {pipeline_name}"
            )
            console.print(
                "Available pipelines: orchestration, document_processing, query_evaluation, help_center"
            )
            return

        # Show success message
        console.print()
        console.print(
            Panel(
                "[bold green]Pipeline execution started successfully![/bold green]\n\n"
                "The pipeline will continue running in the background.\n"
                "You can check the status with the ZenML CLI:\n\n"
                "[dim]zenml pipeline runs list[/dim]\n"
                "[dim]zenml pipeline runs describe <run-id>[/dim]\n",
                title="Success",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error running pipeline:[/bold red] {str(e)}")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@app.command("test-enrichment")
def test_enrichment(
    file_path: str = typer.Argument(..., help="Path to the markdown file to enrich"),
    max_chunks: int = typer.Option(
        5, "--max-chunks", "-n", help="Maximum number of chunks to display"
    ),
):
    """Test contextual enrichment on a markdown file.

    This command processes a markdown file, chunks it, and applies contextual enrichment
    to each chunk, displaying the before and after results.
    """
    import asyncio
    from ...domain.models.document import Document
    from ...infrastructure.document_processing.chunking_service import (
        MarkdownChunkingService,
    )

    async def process_file():
        # Initialize services
        chunking_service = MarkdownChunkingService()
        enrichment_service = container.get("contextual_enrichment_service")

        # Read the markdown file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            return
        except Exception as e:
            console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")
            return

        # Create document object
        doc_title = (
            file_path.split("/")[-1].replace(".md", "").replace("_", " ").title()
        )
        document = Document(
            title=doc_title,
            content=content,
            metadata={"source_path": file_path, "file_type": "markdown"},
        )

        # Create chunks
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking document", total=100)
            progress.update(task, advance=30)

            chunks = await chunking_service.dynamic_chunk_document(
                document=document, min_chunk_size=300, max_chunk_size=800
            )

            progress.update(task, advance=40, description="Enriching chunks")

            # Enrich chunks
            enriched_chunks = await enrichment_service.enrich_chunks(document, chunks)

            progress.update(task, completed=100, description="Complete")

        # Display document info
        console.print()
        console.print(
            Panel(
                f"[bold]Document:[/bold] {document.title}\n"
                f"[bold]Content Length:[/bold] {len(document.content)} characters\n"
                f"[bold]Chunks Created:[/bold] {len(chunks)}",
                title="Document Information",
                border_style="blue",
            )
        )

        # Display results (limit to max_chunks)
        display_chunks = min(len(chunks), max_chunks)
        for i, (original, enriched) in enumerate(
            zip(chunks[:display_chunks], enriched_chunks[:display_chunks])
        ):
            # Calculate the added context (what was prepended)
            added_context = enriched.metadata.get("context", "No context added")

            # Display the chunk comparison
            console.print()
            console.print(
                Panel(
                    f"[bold cyan]Original Content (first 200 chars):[/bold cyan]\n"
                    f"{original.content[:200].strip()}...\n\n"
                    f"[bold green]Enriched Content (first 200 chars):[/bold green]\n"
                    f"{enriched.content[:200].strip()}...\n\n"
                    f"[bold yellow]Added Context:[/bold yellow]\n"
                    f"{added_context}",
                    title=f"Chunk {i+1}/{len(chunks)}",
                    border_style="green",
                )
            )

        # If we limited the display, show a message
        if len(chunks) > max_chunks:
            console.print(
                f"\n[dim]Showing {max_chunks} of {len(chunks)} chunks. Use --max-chunks option to view more.[/dim]"
            )

    # Run the async function
    asyncio.run(process_file())


@app.command("evaluate-enrichment")
def evaluate_enrichment(
    document_path: str = typer.Argument(
        ..., help="Path to the markdown document to evaluate"
    ),
    output_dir: str = typer.Option(
        "data/evaluation",
        "--output-dir",
        "-o",
        help="Directory to save evaluation results",
    ),
    generate_dataset: bool = typer.Option(
        True,
        "--generate-dataset/--use-existing",
        help="Generate a new dataset or use existing one",
    ),
    num_queries: int = typer.Option(
        10, "--num-queries", "-n", help="Number of queries to generate for evaluation"
    ),
    first_stage_k: int = typer.Option(
        20, "--first-stage-k", "-k1", help="Number of results for first stage retrieval"
    ),
    second_stage_k: int = typer.Option(
        5,
        "--second-stage-k",
        "-k2",
        help="Number of results for second stage retrieval",
    ),
):
    """Evaluate the impact of contextual enrichment on retrieval quality.

    This command performs a comprehensive evaluation comparing retrieval performance
    with and without contextual enrichment. It generates test queries, processes
    a document with and without enrichment, and measures recall, precision, and other metrics.
    """
    import os
    import asyncio
    import json
    from pathlib import Path
    from datetime import datetime
    from ...application.pipelines.evaluation.dataset_generator import (
        generate_evaluation_dataset,
    )
    from ...application.pipelines.evaluation.contextual_enrichment_pipeline import (
        load_evaluation_dataset,
        prepare_document_variations,
        evaluate_query,
        analyze_evaluation_results,
    )

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine dataset path
    dataset_path = os.path.join(output_dir, "evaluation_dataset.json")

    async def run_evaluation():
        # Step 1: Generate or load evaluation dataset
        if generate_dataset or not os.path.exists(dataset_path):
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Generating evaluation dataset", total=100)
                progress.update(task, advance=10)

                # Generate dataset
                dataset_path_result = await generate_evaluation_dataset(
                    input_file_path=document_path,
                    output_path=output_dir,
                    num_queries=num_queries,
                    num_relevant_per_query=3,
                )

                progress.update(task, completed=100)

            if not os.path.exists(dataset_path):
                console.print(
                    f"[bold red]Error:[/bold red] Failed to generate dataset at {dataset_path}"
                )
                return

        # Step 2: Load the dataset
        console.print(f"Loading evaluation dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)
        dataset = {"queries": dataset_data}

        # Step 3: Prepare document variations (with and without enrichment)
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing document", total=100)
            progress.update(task, advance=10)

            # Process the documents directly
            import uuid
            from pathlib import Path
            from epic_rag.domain.models.document import Document

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Read document
            with open(document_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create document objects with different IDs
            base_document = Document(
                title=Path(document_path).stem,
                content=content,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={
                    "source": "evaluation",
                    "filename": os.path.basename(document_path),
                    "enriched": False,
                },
            )

            enriched_document = Document(
                title=f"{Path(document_path).stem}_enriched",
                content=content,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={
                    "source": "evaluation",
                    "filename": os.path.basename(document_path),
                    "enriched": True,
                },
            )

            # Get required services
            document_repository = container.get("document_repository")
            vector_repository = container.get("vector_repository")
            chunking_service = container.get("chunking_service")
            embedding_service = container.get("embedding_service")

            # Try to get the contextual enrichment service
            try:
                enrichment_service = container.get("contextual_enrichment_service")
                has_enrichment = True
            except Exception:
                print("Warning: Contextual enrichment service not found in container")
                has_enrichment = False

            # Create use case
            ingest_use_case = IngestDocumentUseCase(
                document_repository=document_repository,
                vector_repository=vector_repository,
                chunking_service=chunking_service,
                embedding_service=embedding_service,
            )

            # Process base document
            print("Processing base document...")
            base_result = await ingest_use_case.execute(
                document=base_document,
                dynamic_chunking=True,
                min_chunk_size=200,
                max_chunk_size=500,
                apply_contextual_enrichment=False,
            )

            # Process enriched document
            print("Processing enriched document...")
            if has_enrichment:
                enriched_result = await ingest_use_case.execute(
                    document=enriched_document,
                    dynamic_chunking=True,
                    min_chunk_size=200,
                    max_chunk_size=500,
                    apply_contextual_enrichment=True,
                )
            else:
                print(
                    "Warning: No enrichment service available, skipping enriched document"
                )
                enriched_result = base_result

            doc_ids = {
                "base_document_id": base_document.id,
                "enriched_document_id": enriched_document.id,
            }

            print(f"Prepared documents for evaluation:")
            print(f"  Base document ID: {base_document.id}")
            print(f"  Enriched document ID: {enriched_document.id}")

            progress.update(task, completed=100)

        # Step 4: Run evaluation queries
        console.print("\nEvaluating queries...")
        results = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            evaluate_task = progress.add_task(
                "Evaluating queries", total=len(dataset["queries"])
            )

            for i, query in enumerate(dataset["queries"]):
                progress.update(
                    evaluate_task,
                    description=f"Evaluating query {i+1}/{len(dataset['queries'])}",
                )

                # Extract query data
                query_text = query["query"]
                relevant_chunk_ids = query.get("relevant_chunk_ids", [])

                # Get services
                embedding_service = container.get("embedding_service")
                retrieval_service = container.get("retrieval_service")

                # Create use case
                retrieve_use_case = RetrieveContextUseCase(
                    embedding_service=embedding_service,
                    retrieval_service=retrieval_service,
                )

                # Process the query against the base document
                base_result = await retrieve_use_case.execute(
                    query_text=query_text,
                    first_stage_k=first_stage_k,
                    second_stage_k=second_stage_k,
                    min_relevance_score=0.0,  # No filtering for evaluation
                    use_query_transformation=False,  # Raw query
                    merge_related_chunks=False,  # No merging for fair comparison
                    document_filter={"id": doc_ids["base_document_id"]},
                    max_results=100,
                )

                # Process the query against the enriched document
                enriched_result = await retrieve_use_case.execute(
                    query_text=query_text,
                    first_stage_k=first_stage_k,
                    second_stage_k=second_stage_k,
                    min_relevance_score=0.0,  # No filtering for evaluation
                    use_query_transformation=False,  # Raw query
                    merge_related_chunks=False,  # No merging for fair comparison
                    document_filter={"id": doc_ids["enriched_document_id"]},
                    max_results=100,
                )

                # Extract retrieved chunks
                base_retrieved_ids = [
                    chunk.id for chunk in base_result.first_stage_results.chunks
                ]
                enriched_retrieved_ids = [
                    chunk.id for chunk in enriched_result.first_stage_results.chunks
                ]

                # Calculate metrics
                from epic_rag.application.pipelines.evaluation.metrics import (
                    calculate_retrieval_metrics,
                )

                base_metrics = calculate_retrieval_metrics(
                    retrieved_ids=base_retrieved_ids,
                    relevant_ids=relevant_chunk_ids,
                )

                enriched_metrics = calculate_retrieval_metrics(
                    retrieved_ids=enriched_retrieved_ids,
                    relevant_ids=relevant_chunk_ids,
                )

                # Prepare result
                result = {
                    "query": query_text,
                    "relevant_chunks": relevant_chunk_ids,
                    "base_metrics": base_metrics.as_dict(),
                    "enriched_metrics": enriched_metrics.as_dict(),
                    "base_retrieved": base_retrieved_ids[
                        :10
                    ],  # Include first 10 for inspection
                    "enriched_retrieved": enriched_retrieved_ids[
                        :10
                    ],  # Include first 10 for inspection
                    "base_latency_ms": base_result.total_latency_ms,
                    "enriched_latency_ms": enriched_result.total_latency_ms,
                }

                # Calculate improvement
                if len(relevant_chunk_ids) > 0:
                    base_recall_20 = base_metrics.recall_at_k.get(20, 0.0)
                    enriched_recall_20 = enriched_metrics.recall_at_k.get(20, 0.0)

                    # Anthropic's metric: failure rate reduction at 20
                    # 1 - recall@20 = failure rate
                    base_failure_rate = 1 - base_recall_20
                    enriched_failure_rate = 1 - enriched_recall_20

                    if base_failure_rate > 0:
                        failure_rate_reduction = (
                            base_failure_rate - enriched_failure_rate
                        ) / base_failure_rate
                    else:
                        failure_rate_reduction = 0.0

                    result["improvement"] = {
                        "recall_at_20_absolute": enriched_recall_20 - base_recall_20,
                        "recall_at_20_relative": (
                            (enriched_recall_20 / base_recall_20) - 1
                            if base_recall_20 > 0
                            else float("inf")
                        ),
                        "failure_rate_reduction": failure_rate_reduction,
                    }

                results.append(result)
                progress.update(evaluate_task, advance=1)

        # Step 5: Analyze results
        output_path = os.path.join(output_dir, "evaluation_results.json")
        console.print("\nAnalyzing results...")

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert individual metrics to RetrievalMetrics objects
        from epic_rag.application.pipelines.evaluation.metrics import (
            RetrievalMetrics,
            calculate_aggregate_metrics,
        )

        base_metrics_list = []
        enriched_metrics_list = []

        for result in results:
            # Base metrics
            base_recall = result["base_metrics"]["recall_at_k"]
            base_precision = result["base_metrics"]["precision_at_k"]
            base_ndcg = result["base_metrics"]["ndcg_at_k"]
            base_mrr = result["base_metrics"]["mean_reciprocal_rank"]

            # Convert string keys to integers
            base_recall = {int(k): v for k, v in base_recall.items()}
            base_precision = {int(k): v for k, v in base_precision.items()}
            base_ndcg = {int(k): v for k, v in base_ndcg.items()}

            base_metrics = RetrievalMetrics(
                recall_at_k=base_recall,
                precision_at_k=base_precision,
                ndcg_at_k=base_ndcg,
                mean_reciprocal_rank=base_mrr,
            )
            base_metrics_list.append(base_metrics)

            # Enriched metrics
            enriched_recall = result["enriched_metrics"]["recall_at_k"]
            enriched_precision = result["enriched_metrics"]["precision_at_k"]
            enriched_ndcg = result["enriched_metrics"]["ndcg_at_k"]
            enriched_mrr = result["enriched_metrics"]["mean_reciprocal_rank"]

            # Convert string keys to integers
            enriched_recall = {int(k): v for k, v in enriched_recall.items()}
            enriched_precision = {int(k): v for k, v in enriched_precision.items()}
            enriched_ndcg = {int(k): v for k, v in enriched_ndcg.items()}

            enriched_metrics = RetrievalMetrics(
                recall_at_k=enriched_recall,
                precision_at_k=enriched_precision,
                ndcg_at_k=enriched_ndcg,
                mean_reciprocal_rank=enriched_mrr,
            )
            enriched_metrics_list.append(enriched_metrics)

        # Calculate aggregate metrics
        aggregate_base = calculate_aggregate_metrics(base_metrics_list)
        aggregate_enriched = calculate_aggregate_metrics(enriched_metrics_list)

        # Calculate improvement metrics
        improvements = {}
        for k in sorted(aggregate_base.recall_at_k.keys()):
            base_val = aggregate_base.recall_at_k.get(k, 0.0)
            enriched_val = aggregate_enriched.recall_at_k.get(k, 0.0)
            abs_diff = enriched_val - base_val
            rel_diff = (enriched_val / base_val) - 1 if base_val > 0 else float("inf")

            improvements[f"recall@{k}"] = {
                "base": base_val,
                "enriched": enriched_val,
                "absolute_improvement": abs_diff,
                "relative_improvement": rel_diff,
            }

        # Calculate Anthropic's metric: failure rate reduction at 20
        base_failure_rate = 1 - aggregate_base.recall_at_k.get(20, 0.0)
        enriched_failure_rate = 1 - aggregate_enriched.recall_at_k.get(20, 0.0)

        if base_failure_rate > 0:
            failure_rate_reduction = (
                base_failure_rate - enriched_failure_rate
            ) / base_failure_rate
        else:
            failure_rate_reduction = 0.0

        # Calculate average latency
        avg_base_latency = (
            sum(r["base_latency_ms"] for r in results) / len(results) if results else 0
        )
        avg_enriched_latency = (
            sum(r["enriched_latency_ms"] for r in results) / len(results)
            if results
            else 0
        )

        # Prepare summary
        summary = {
            "total_queries": len(dataset["queries"]),
            "base_metrics": aggregate_base.as_dict(),
            "enriched_metrics": aggregate_enriched.as_dict(),
            "improvements": improvements,
            "anthropic_metric": {
                "base_failure_rate": base_failure_rate,
                "enriched_failure_rate": enriched_failure_rate,
                "failure_rate_reduction": failure_rate_reduction,
            },
            "latency": {
                "base_avg_ms": avg_base_latency,
                "enriched_avg_ms": avg_enriched_latency,
                "overhead_ms": avg_enriched_latency - avg_base_latency,
                "overhead_percent": (
                    ((avg_enriched_latency / avg_base_latency) - 1) * 100
                    if avg_base_latency > 0
                    else 0
                ),
            },
        }

        # Save detailed results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "individual_results": results,
                },
                f,
                indent=2,
            )

        print(f"Evaluation complete. Report saved to {output_path}")

        # Step 6: Display results
        base_recall_20 = summary["base_metrics"]["recall_at_k"]["20"]
        enriched_recall_20 = summary["enriched_metrics"]["recall_at_k"]["20"]
        failure_rate_reduction = summary["anthropic_metric"]["failure_rate_reduction"]

        console.print()
        console.print(
            Panel(
                f"[bold cyan]Base Recall@20:[/bold cyan] {base_recall_20:.4f}\n"
                f"[bold green]Enriched Recall@20:[/bold green] {enriched_recall_20:.4f}\n"
                f"[bold cyan]Base Failure Rate:[/bold cyan] {1 - base_recall_20:.4f}\n"
                f"[bold green]Enriched Failure Rate:[/bold green] {1 - enriched_recall_20:.4f}\n"
                f"[bold yellow]Failure Rate Reduction:[/bold yellow] {failure_rate_reduction:.2%}\n\n"
                f"[bold]Anthropic's Key Metric:[/bold] Failure Rate Reduction\n"
                f"Failure Rate = 1 - Recall@20\n"
                f"Reduction = (Base Failure Rate - Enriched Failure Rate) / Base Failure Rate",
                title="Evaluation Results",
                border_style="green",
            )
        )

        # Display latency impact
        latency_overhead_ms = summary["latency"]["overhead_ms"]
        latency_overhead_pct = summary["latency"]["overhead_percent"]

        console.print(
            Panel(
                f"[bold cyan]Base Latency:[/bold cyan] {summary['latency']['base_avg_ms']:.2f}ms\n"
                f"[bold green]Enriched Latency:[/bold green] {summary['latency']['enriched_avg_ms']:.2f}ms\n"
                f"[bold yellow]Overhead:[/bold yellow] {latency_overhead_ms:.2f}ms (+{latency_overhead_pct:.1f}%)",
                title="Performance Impact",
                border_style="blue",
            )
        )

        console.print(f"\nDetailed results saved to: [bold]{output_path}[/bold]")

    # Run the evaluation
    asyncio.run(run_evaluation())


@app.command("info")
def show_system_info():
    """Show system information and statistics."""
    # Get repositories
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")

    # Try to get embedding service with cache stats
    embedding_service = container.get("embedding_service")
    has_cache_stats = hasattr(embedding_service, "get_cache_stats")

    # Get statistics async
    async def get_stats():
        try:
            # Get repository statistics
            db_stats = await document_repository.get_statistics()

            # Get collection stats
            vector_stats = await vector_repository.get_collection_stats()

            # Get cache stats if available
            cache_stats = None
            if has_cache_stats:
                cache_stats = await embedding_service.get_cache_stats()

            return {
                "db_stats": db_stats,
                "vector_stats": vector_stats,
                "cache_stats": cache_stats,
            }
        except Exception as e:
            console.print(f"[bold red]Error getting statistics:[/bold red] {str(e)}")
            return {"db_stats": {}, "vector_stats": {}, "cache_stats": None}

    # Run the stats collection
    stats = asyncio.run(get_stats())

    # Show system info
    cache_info = ""
    if settings.embedding.cache.enabled:
        cache_status = "Enabled"

        # Add cache stats if available
        if stats.get("cache_stats"):
            cache_stats = stats["cache_stats"]
            cache_entries = cache_stats.get("total_entries", 0)
            cache_size = cache_stats.get("db_size_bytes", 0) / (1024 * 1024)
            cache_info = f"\n[cyan]Embedding Cache:[/cyan] {cache_status} ({cache_entries} entries, {cache_size:.2f} MB)"
        else:
            cache_info = f"\n[cyan]Embedding Cache:[/cyan] {cache_status}"
    else:
        cache_info = "\n[cyan]Embedding Cache:[/cyan] Disabled"

    # Get appropriate model name based on provider
    model_name = settings.embedding.model
    if settings.embedding.provider.lower() == "openai":
        model_name = settings.embedding.openai_model
    elif settings.embedding.provider.lower() == "gemini":
        model_name = settings.embedding.gemini_model
    elif settings.embedding.provider.lower() == "huggingface":
        model_name = settings.embedding.huggingface_model

    # Get reranker status
    reranker_info = ""
    if settings.retrieval.reranker.enabled:
        reranker_status = "Enabled"
        reranker_model = settings.retrieval.reranker.model_name
        reranker_info = f"\n[cyan]Reranker:[/cyan] {reranker_status} / {reranker_model}"
    else:
        reranker_info = "\n[cyan]Reranker:[/cyan] Disabled"

    # Get ZenML info
    try:
        import subprocess

        zenml_version = subprocess.check_output(["zenml", "version"], text=True).strip()
        zenml_info = f"\n[cyan]ZenML:[/cyan] {zenml_version}"
    except Exception:
        zenml_info = "\n[cyan]ZenML:[/cyan] Installed"

    console.print(
        Panel(
            f"[bold]Epic Documentation RAG System[/bold]\n"
            f"Using Anthropic's Contextual Retrieval methodology\n\n"
            f"[cyan]Environment:[/cyan] {settings.environment}\n"
            f"[cyan]Database:[/cyan] {settings.database.path}\n"
            f"[cyan]Vector Database:[/cyan] {settings.qdrant.url or 'Local Qdrant'}\n"
            f"[cyan]Embedding Model:[/cyan] {settings.embedding.provider} / {model_name}\n"
            f"[cyan]LLM Model:[/cyan] {settings.llm.provider} / {settings.llm.model}"
            f"{cache_info}"
            f"{reranker_info}"
            f"{zenml_info}",
            title="System Information",
            border_style="blue",
        )
    )

    # Show statistics
    console.print()
    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add document stats
    db_stats = stats["db_stats"]
    if "document_count" in db_stats:
        table.add_row("Documents Stored", str(db_stats["document_count"]["value"]))

    if "chunk_count" in db_stats:
        table.add_row("Chunks Stored", str(db_stats["chunk_count"]["value"]))

    if "total_content_size" in db_stats:
        size_kb = db_stats["total_content_size"]["value"] / 1024
        table.add_row("Total Content Size", f"{size_kb:.2f} KB")

    if "avg_chunk_size" in db_stats:
        table.add_row(
            "Average Chunk Size", f"{db_stats['avg_chunk_size']['value']} chars"
        )

    # Add vector stats if available
    vector_stats = stats["vector_stats"]
    if "vector_count" in vector_stats:
        table.add_row("Vectors Stored", str(vector_stats["vector_count"]))

    if "segment_count" in vector_stats:
        table.add_row("Vector Segments", str(vector_stats["segment_count"]))

    if "size_bytes" in vector_stats and vector_stats["size_bytes"]:
        size_mb = vector_stats["size_bytes"] / (1024 * 1024)
        table.add_row("Vector DB Size", f"{size_mb:.2f} MB")

    console.print(table)


@db_app.command("info")
def db_info():
    """Show detailed information about the database."""

    # Create a function that will run the async code properly
    async def _show_info():
        document_repository = container.get("document_repository")
        stats = await document_repository.get_statistics()

        # Create a table to display stats
        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Last Updated", style="blue")

        for key, data in stats.items():
            # Format values for better readability
            if key == "database_size" and "value" in data:
                # Convert bytes to KB/MB/GB as appropriate
                size_bytes = data["value"]
                if size_bytes < 1024:
                    formatted_value = f"{size_bytes} bytes"
                elif size_bytes < 1024 * 1024:
                    formatted_value = f"{size_bytes / 1024:.2f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    formatted_value = f"{size_bytes / (1024 * 1024):.2f} MB"
                else:
                    formatted_value = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
            else:
                formatted_value = str(data["value"])

            table.add_row(
                key.replace("_", " ").title(),
                formatted_value,
                data.get("updated_at", "N/A"),
            )

        console.print(table)

        # Database file info
        try:
            db_path = document_repository.db_path
            if os.path.exists(db_path):
                db_info_panel = Panel(
                    f"Path: [cyan]{db_path}[/cyan]\n"
                    f"Size: [green]{os.path.getsize(db_path) / (1024 * 1024):.2f} MB[/green]\n"
                    f"Created: [blue]{datetime.datetime.fromtimestamp(os.path.getctime(db_path))}[/blue]\n"
                    f"Modified: [blue]{datetime.datetime.fromtimestamp(os.path.getmtime(db_path))}[/blue]",
                    title="Database File Information",
                    expand=False,
                )
                console.print(db_info_panel)
        except Exception as e:
            console.print(f"[red]Error getting database file info: {str(e)}[/red]")

    # Simplify the async execution to use asyncio.run directly
    def run_db_info():
        asyncio.run(_show_info())

    # Run the function
    run_db_info()


@db_app.command("cleanup-orphans")
def cleanup_orphaned_chunks():
    """Clean up orphaned chunks (chunks without a parent document)."""

    async def _cleanup():
        document_repository = container.get("document_repository")

        with console.status("[bold green]Finding orphaned chunks..."):
            orphaned_chunks = await document_repository.find_orphaned_chunks()

        if not orphaned_chunks:
            console.print("[green]No orphaned chunks found![/green]")
            return

        if not Confirm.ask(
            f"Found {len(orphaned_chunks)} orphaned chunks. Delete them?"
        ):
            console.print("[yellow]Operation canceled.[/yellow]")
            return

        with console.status("[bold green]Deleting orphaned chunks..."):
            deleted_count = await document_repository.delete_orphaned_chunks()

        console.print(
            f"[green]Successfully deleted {deleted_count} orphaned chunks[/green]"
        )

    # Simplify the async execution to use asyncio.run directly
    def run_cleanup():
        asyncio.run(_cleanup())

    run_cleanup()


@db_app.command("backup")
def backup_database(
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for backup file"
    )
):
    """Backup the SQLite database."""

    async def _backup():
        document_repository = container.get("document_repository")
        db_path = document_repository.db_path

        # Generate backup filename if not provided
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.dirname(db_path)
            output_filename = f"{os.path.basename(db_path)}_{timestamp}.backup"
            backup_path = os.path.join(output_dir, output_filename)
        else:
            backup_path = output_path

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)

        console.print(f"[bold]Backing up database to:[/bold] {backup_path}")

        try:
            # Create a backup using SQLite's backup API
            async with aiosqlite.connect(db_path) as source_db:
                async with aiosqlite.connect(backup_path) as dest_db:
                    with console.status("[bold green]Creating backup..."):
                        await source_db.backup(dest_db)

            # Verify backup was created
            if os.path.exists(backup_path):
                source_size = os.path.getsize(db_path)
                backup_size = os.path.getsize(backup_path)
                console.print(f"[green]Backup completed successfully![/green]")
                console.print(
                    f"Source database size: {source_size / (1024 * 1024):.2f} MB"
                )
                console.print(
                    f"Backup database size: {backup_size / (1024 * 1024):.2f} MB"
                )
            else:
                console.print(
                    "[bold red]Backup file was not created properly[/bold red]"
                )
        except Exception as e:
            console.print(
                f"[bold red]Error backing up database:[/bold red] {str(e)}"
            )

    # Simplify the async execution to use asyncio.run directly
    def run_backup():
        asyncio.run(_backup())

    run_backup()


@db_app.command("vacuum")
def vacuum_database():
    """Run vacuum operation on the SQLite database to reclaim unused space."""

    async def _vacuum():
        document_repository = container.get("document_repository")

        if not Confirm.ask(
            "This operation may take some time for large databases. Continue?"
        ):
            console.print("[yellow]Operation canceled.[/yellow]")
            return

        console.print("[bold]Running VACUUM operation...[/bold]")
        start_time = time.time()

        with console.status(
            "[bold green]Vacuuming database (this may take a while)..."
        ):
            size_before, size_after = await document_repository.vacuum_database()

        elapsed_time = time.time() - start_time
        console.print(
            f"[green]VACUUM completed in {elapsed_time:.2f} seconds[/green]"
        )
        console.print(f"Database size before: {size_before:.2f} MB")
        console.print(f"Database size after: {size_after:.2f} MB")

        if size_before > 0:  # Avoid division by zero
            space_saved = size_before - size_after
            percent_saved = (space_saved / size_before) * 100
            console.print(
                f"Space saved: {space_saved:.2f} MB ({percent_saved:.2f}%)"
            )

    # Simplify the async execution to use asyncio.run directly
    def run_vacuum():
        asyncio.run(_vacuum())

    run_vacuum()


@db_app.command("inspect-document")
def inspect_document(
    document_id: Optional[str] = typer.Option(
        None, "--id", help="Document ID to inspect"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", help="Document title to search for"
    ),
    epic_page_id: Optional[str] = typer.Option(
        None, "--epic-id", help="Epic page ID to search for"
    ),
    show_chunks: bool = typer.Option(
        False, "--chunks", "-c", help="Show document chunks"
    ),
    show_metadata: bool = typer.Option(
        False, "--metadata", "-m", help="Show detailed metadata"
    ),
):
    """Inspect a document and its metadata."""
    # Use the script we created for this purpose
    import sys
    import subprocess
    
    cmd = [
        sys.executable, 
        "db_inspect_document.py"
    ]
    
    # Add arguments
    if document_id:
        cmd.extend(["--id", document_id])
    if title:
        cmd.extend(["--title", title])
    if epic_page_id:
        cmd.extend(["--epic-id", epic_page_id])
    if show_chunks:
        cmd.append("--chunks")
    if show_metadata:
        cmd.append("--metadata")
    
    # Execute the script
    subprocess.run(cmd)


@db_app.command("list-documents")
def list_documents(
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of documents to list"
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset to start listing from"
    ),
    sort_by: str = typer.Option(
        "updated_at", "--sort", "-s", help="Sort field (title, created_at, updated_at)"
    ),
    descending: bool = typer.Option(
        True, "--desc/--asc", help="Sort in descending order"
    ),
):
    """List documents in the database."""
    # Use the script we created for this purpose
    import sys
    import subprocess
    
    cmd = [
        sys.executable, 
        "db_list_documents.py"
    ]
    
    # Add arguments
    if limit != 20:  # Only add if not default
        cmd.extend(["--limit", str(limit)])
    if offset != 0:  # Only add if not default
        cmd.extend(["--offset", str(offset)])
    
    # Execute the script
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
