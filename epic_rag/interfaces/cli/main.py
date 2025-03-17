"""Command-line interface for Epic Documentation RAG system."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from ...domain.models.document import Document
from ...infrastructure.config.settings import settings
from ...infrastructure.container import container, setup_container
from ...application.use_cases.ingest_document import IngestDocumentUseCase
from ...application.use_cases.retrieve_context import RetrieveContextUseCase

# Create Typer app
app = typer.Typer(
    name="epic-rag",
    help="Epic Documentation RAG System",
    add_completion=False,
)

# Rich console for pretty output
console = Console()


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
):
    """Test BM25 search functionality."""
    from epic_rag.domain.models.retrieval import Query

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

            # Run BM25 search
            progress.update(task, advance=30, description="Searching")
            result = await bm25_service.search(query, limit=limit)

            progress.update(task, completed=100, description="Complete")
            return result

    # Run search
    try:
        result = asyncio.run(run_search())

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
):
    """Test hybrid search with BM25 and vector search with rank fusion."""
    from epic_rag.domain.models.retrieval import Query, RetrievalResult

    # Get required services
    bm25_service = container.get("bm25_search_service")
    embedding_service = container.get("embedding_service")
    vector_repository = container.get("vector_repository")
    document_repository = container.get("document_repository")
    rank_fusion_service = container.get("rank_fusion_service")

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
            progress.update(task, advance=20, description="Fusing results")
            fused_results = await rank_fusion_service.fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
            )

            progress.update(task, completed=100, description="Complete")
            return {
                "vector": vector_results,
                "bm25": bm25_results,
                "fused": fused_results,
            }

    # Run search
    try:
        results = asyncio.run(run_search())

        # Display results
        console.print()
        console.print(
            Panel(
                f"[bold]Query:[/bold] {query_text}\n"
                f"[bold]BM25 Results:[/bold] {len(results['bm25'].chunks)}\n"
                f"[bold]Vector Results:[/bold] {len(results['vector'].chunks)}\n"
                f"[bold]Fused Results:[/bold] {len(results['fused'].chunks)}\n"
                f"[bold]BM25 Weight:[/bold] {bm25_weight}\n"
                f"[bold]Vector Weight:[/bold] {vector_weight}",
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

        # Show fused results
        console.print()
        console.print("[bold blue]Fused Results:[/bold blue]")
        for i, chunk in enumerate(results["fused"].chunks, 1):
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

            console.print(
                Panel(
                    f"[bold]Fused Score:[/bold] {chunk.relevance_score:.4f}\n"
                    f"[bold]BM25 Rank:[/bold] {bm25_rank}\n"
                    f"[bold]Vector Rank:[/bold] {vector_rank}\n"
                    f"[bold]Document:[/bold] {chunk.metadata.get('title', 'Untitled')}\n"
                    f"[bold]Content:[/bold]\n{content}",
                    title=f"Result {i}/{len(results['fused'].chunks)}",
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

    console.print(
        Panel(
            f"[bold]Epic Documentation RAG System[/bold]\n"
            f"Using Anthropic's Contextual Retrieval methodology\n\n"
            f"[cyan]Environment:[/cyan] {settings.environment}\n"
            f"[cyan]Database:[/cyan] {settings.database.path}\n"
            f"[cyan]Vector Database:[/cyan] {settings.qdrant.url or 'Local Qdrant'}\n"
            f"[cyan]Embedding Model:[/cyan] {settings.embedding.provider} / {model_name}\n"
            f"[cyan]LLM Model:[/cyan] {settings.llm.provider} / {settings.llm.model}"
            f"{cache_info}",
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


if __name__ == "__main__":
    app()
