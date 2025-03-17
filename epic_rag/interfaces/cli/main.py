"""Command-line interface for Epic Documentation RAG system."""

import os
import asyncio
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from ...domain.models.document import Document
from ...domain.models.retrieval import Query
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


@app.command("info")
def show_system_info():
    """Show system information and statistics."""
    # Get repositories
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")

    # Get statistics async
    async def get_stats():
        try:
            # Get document counts
            documents = await document_repository.list_documents()
            doc_count = len(documents)

            # Get collection stats
            vector_stats = await vector_repository.get_collection_stats()

            return {"document_count": doc_count, "vector_stats": vector_stats}
        except Exception as e:
            console.print(f"[bold red]Error getting statistics:[/bold red] {str(e)}")
            return {"document_count": 0, "vector_stats": {}}

    # Run the stats collection
    stats = asyncio.run(get_stats())

    # Show system info
    console.print(
        Panel(
            f"[bold]Epic Documentation RAG System[/bold]\n"
            f"Using Anthropic's Contextual Retrieval methodology\n\n"
            f"[cyan]Environment:[/cyan] {settings.environment}\n"
            f"[cyan]Database:[/cyan] {settings.database.path}\n"
            f"[cyan]Vector Database:[/cyan] {settings.qdrant.url or 'Local Qdrant'}\n"
            f"[cyan]Embedding Model:[/cyan] {settings.embedding.provider} / {settings.embedding.model}\n"
            f"[cyan]LLM Model:[/cyan] {settings.llm.provider} / {settings.llm.model}\n",
            title="System Information",
            border_style="blue",
        )
    )

    # Show statistics
    console.print()
    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Documents Stored", str(stats["document_count"]))

    # Add vector stats if available
    if "vector_count" in stats["vector_stats"]:
        table.add_row("Vectors Stored", str(stats["vector_stats"]["vector_count"]))

    if "segment_count" in stats["vector_stats"]:
        table.add_row("Vector Segments", str(stats["vector_stats"]["segment_count"]))

    console.print(table)


if __name__ == "__main__":
    app()
