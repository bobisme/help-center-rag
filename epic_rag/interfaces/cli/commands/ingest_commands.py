"""Commands for the ingestion pipeline."""

import asyncio
import json
import os
import typer
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from ....domain.models.document import Document
from ....infrastructure.container import container, setup_container
from ....application.use_cases.ingest_document import IngestDocumentUseCase
from .common import console, create_progress_bar

ingest_app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Document ingestion pipeline commands for inspecting the processing stages",
)


def load_document_from_json(index: int) -> Document:
    """Load a document from the epic-docs.json file by index.

    Args:
        index: The index of the document to load

    Returns:
        The loaded document
    """
    json_path = "output/epic-docs.json"
    if not os.path.exists(json_path):
        console.print(f"[bold red]File not found: {json_path}[/bold red]")
        raise typer.Exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "pages" not in data or not isinstance(data["pages"], list):
            console.print(
                f"[bold red]Invalid format: {json_path} is not a consolidated file[/bold red]"
            )
            raise typer.Exit(1)

        if index >= len(data["pages"]):
            console.print(
                f"[bold red]Index out of range: {index} (max: {len(data['pages'])-1})[/bold red]"
            )
            raise typer.Exit(1)

        page = data["pages"][index]

        # Extract document data
        title = page.get("title", f"Untitled_Page_{index}")
        page_id = index  # Use index as ID
        category = page.get("metadata", {}).get("path", ["Uncategorized"])[0]
        updated_at = page.get("metadata", {}).get("crawlDate")

        # Convert HTML content to markdown (assuming it has already been converted)
        content = page.get("content", "")
        if not content and "rawHtml" in page:
            from html2md import convert_html_to_markdown, preprocess_html
            
            # Define images directory - our default should be output/images
            images_dir = "output/images"
            if os.path.exists(images_dir):
                console.print(f"[green]Using images from {images_dir}[/green]")
            else:
                console.print(f"[yellow]Warning: Images directory {images_dir} not found. Images will be removed.[/yellow]")

            html = preprocess_html(page["rawHtml"], images_dir)
            content = convert_html_to_markdown(html, images_dir=images_dir)

        # Check if content already starts with the title as a heading
        has_title_heading = False
        if content:
            # Check for title in various heading formats
            heading_pattern = f"# {title}"
            has_title_heading = content.strip().startswith(heading_pattern)

        # Add title heading only if not already present
        final_content = content
        if not has_title_heading:
            final_content = f"# {title}\n\n{content}"

        # Create document
        document = Document(
            title=title,
            content=final_content,
            epic_page_id=page_id,
            metadata={
                "category": category,
                "updated_at": updated_at,
                "source_path": json_path,
                "page_index": index,
            },
        )

        return document

    except Exception as e:
        console.print(f"[bold red]Error loading document: {str(e)}[/bold red]")
        raise typer.Exit(1)


@ingest_app.command("show-doc")
def print_document(
    index: int = typer.Option(
        ..., "--index", "-i", help="Document index in epic-docs.json"
    )
):
    """Print the document at the specified index from epic-docs.json."""
    document = load_document_from_json(index)

    # Display document info
    console.print(
        Panel(
            f"[bold]Document Information[/bold]\n\n"
            f"Title: [cyan]{document.title}[/cyan]\n"
            f"Epic Page ID: {document.epic_page_id}\n"
            f"Category: {document.metadata.get('category', 'N/A')}\n"
            f"Updated At: {document.metadata.get('updated_at', 'N/A')}\n",
            title="Document Details",
            border_style="green",
        )
    )

    # Display document content
    console.print("\n[bold]Document Content:[/bold]")
    console.print(
        Syntax(document.content, "markdown", theme="monokai", line_numbers=True)
    )


@ingest_app.command("show-chunks")
def show_chunks(
    index: int = typer.Option(
        ..., "--index", "-i", help="Document index in epic-docs.json"
    ),
    with_context: bool = typer.Option(
        False, "--with-context", help="Show chunks with enriched context"
    ),
    with_image_descriptions: bool = typer.Option(
        False, "--with-image-descriptions", help="Show chunks with image descriptions"
    ),
):
    """Show the chunks for a document at the specified index."""
    # Load the document
    document = load_document_from_json(index)

    # Initialize container
    setup_container()

    # Get services
    chunking_service = container.get("chunking_service")
    contextual_enrichment_service = container.get("contextual_enrichment_service")

    async def process_chunks():
        # Chunk the document
        chunks = await chunking_service.dynamic_chunk_document(
            document=document,
            min_chunk_size=300,
            max_chunk_size=800,
        )

        # Apply contextual enrichment if requested
        if with_context and contextual_enrichment_service:
            console.print("[yellow]Applying contextual enrichment...[/yellow]")
            chunks = await contextual_enrichment_service.enrich_chunks(
                document=document, chunks=chunks
            )

        # Apply image descriptions if requested
        if with_image_descriptions:
            image_description_service = container.get("image_description_service")
            console.print("[yellow]Adding image descriptions...[/yellow]")
            for chunk in chunks:
                # Process image descriptions (if the service supports it)
                if hasattr(image_description_service, "process_chunk_images"):
                    chunk = await image_description_service.process_chunk_images(chunk)

        return chunks

    # Process chunks
    chunks = asyncio.run(process_chunks())

    # Display chunks
    console.print(f"\n[bold]Chunks ([cyan]{len(chunks)}[/cyan]):[/bold]")

    for i, chunk in enumerate(chunks):
        console.print(f"\n[bold cyan]Chunk {i+1}[/bold cyan]")
        console.print(f"ID: {chunk.id}")
        console.print(f"Chunk Index: {chunk.chunk_index}")

        if hasattr(chunk, "token_count") and chunk.token_count:
            console.print(f"Token Count: {chunk.token_count}")

        if chunk.metadata:
            console.print("\n[bold]Metadata:[/bold]")
            for key, value in chunk.metadata.items():
                console.print(f"  {key}: {value}")

        console.print("\n[bold]Content:[/bold]")
        console.print(Syntax(chunk.content, "markdown", theme="monokai"))

        # If the chunk has a context field from enrichment, show it
        if with_context and hasattr(chunk, "context") and chunk.context:
            console.print("\n[bold yellow]Context:[/bold yellow]")
            console.print(chunk.context)


@ingest_app.command("embed")
def embed_document(
    index: int = typer.Option(
        ..., "--index", "-i", help="Document index in epic-docs.json"
    ),
):
    """Embed chunks for a document at the specified index and print vectors."""
    # Load the document
    document = load_document_from_json(index)

    # Initialize container
    setup_container()

    # Get services
    chunking_service = container.get("chunking_service")
    contextual_enrichment_service = container.get("contextual_enrichment_service")
    embedding_service = container.get("embedding_service")

    async def process_embeddings():
        # Chunk the document
        console.print("[yellow]Chunking document...[/yellow]")
        chunks = await chunking_service.dynamic_chunk_document(
            document=document,
            min_chunk_size=300,
            max_chunk_size=800,
        )

        # Apply contextual enrichment
        console.print("[yellow]Applying contextual enrichment...[/yellow]")
        chunks = await contextual_enrichment_service.enrich_chunks(
            document=document, chunks=chunks
        )

        # Add image descriptions
        image_description_service = container.get("image_description_service")
        if hasattr(image_description_service, "process_chunk_images"):
            console.print("[yellow]Adding image descriptions...[/yellow]")
            for i, chunk in enumerate(chunks):
                chunks[i] = await image_description_service.process_chunk_images(chunk)

        # Generate embeddings
        console.print("[yellow]Generating embeddings...[/yellow]")
        embedded_chunks = await embedding_service.batch_embed_chunks(chunks)

        return embedded_chunks

    # Process embeddings
    embedded_chunks = asyncio.run(process_embeddings())

    # Display results
    console.print(
        f"\n[bold green]Successfully embedded {len(embedded_chunks)} chunks[/bold green]"
    )

    # Display embedding info for each chunk
    for i, chunk in enumerate(embedded_chunks):
        console.print(f"\n[bold cyan]Chunk {i+1}[/bold cyan]")
        console.print(f"Content Preview: {chunk.content[:50]}...")

        if hasattr(chunk, "embedding") and chunk.embedding:
            vector = chunk.embedding
            console.print(f"\n[bold]Embedding Vector (dim={len(vector)}):[/bold]")
            console.print(f"First 5 values: {vector[:5]}")
            console.print(f"Last 5 values: {vector[-5:]}")


@ingest_app.command("pipeline")
def run_pipeline(
    index: int = typer.Option(
        ..., "--index", "-i", help="Document index in epic-docs.json"
    ),
    no_enrich: bool = typer.Option(False, "--no-enrich", help="Skip contextual enrichment"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Process without saving to database"),
):
    """Run the full ingestion pipeline for a document at the specified index."""
    # Load the document
    document = load_document_from_json(index)

    # Initialize container
    setup_container()

    # Get the required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    chunking_service = container.get("chunking_service")
    embedding_service = container.get("embedding_service")
    contextual_enrichment_service = container.get("contextual_enrichment_service")

    # Create the ingest use case
    ingest_use_case = IngestDocumentUseCase(
        document_repository=document_repository,
        vector_repository=vector_repository,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        contextual_enrichment_service=contextual_enrichment_service,
    )

    # Display what we're about to do
    mode_text = ""
    if dry_run:
        mode_text = " [yellow](DRY RUN - No database changes)[/yellow]"
    if no_enrich:
        mode_text += " [yellow](Without enrichment)[/yellow]"
        
    console.print(
        Panel(
            f"[bold]Running Full Ingestion Pipeline{mode_text}[/bold]\n\n"
            f"Document: [cyan]{document.title}[/cyan]\n"
            f"ID: {document.epic_page_id}\n"
            f"Category: {document.metadata.get('category', 'N/A')}\n",
            title="Pipeline Execution",
            border_style="green",
        )
    )

    # Run the pipeline
    async def run():
        # Execute the use case
        result = await ingest_use_case.execute(
            document=document,
            dynamic_chunking=True,
            min_chunk_size=300,
            max_chunk_size=800,
            chunk_overlap=50,
            apply_enrichment=not no_enrich,
            dry_run=dry_run,
        )

        return result

    # Execute the pipeline
    result = asyncio.run(run())

    # Display results
    console.print(f"\n[bold green]Successfully processed document:[/bold green]")
    console.print(f"Title: {result.title}")
    console.print(f"ID: {result.id}")
    console.print(f"Chunks: {len(result.chunks)}")
    
    if dry_run:
        console.print(
            f"[yellow]Dry run completed - no data was saved to the database.[/yellow]"
        )
    else:
        console.print(
            f"Document stored in SQLite database and vector embeddings stored in Qdrant."
        )


@ingest_app.command("show-raw")
def show_raw_html(
    index: int = typer.Option(
        ..., "--index", "-i", help="Document index in epic-docs.json"
    ),
):
    """Show the raw HTML content for a document at the specified index."""
    json_path = "output/epic-docs.json"
    if not os.path.exists(json_path):
        console.print(f"[bold red]File not found: {json_path}[/bold red]")
        raise typer.Exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "pages" not in data or not isinstance(data["pages"], list):
            console.print(
                f"[bold red]Invalid format: {json_path} is not a consolidated file[/bold red]"
            )
            raise typer.Exit(1)

        if index >= len(data["pages"]):
            console.print(
                f"[bold red]Index out of range: {index} (max: {len(data['pages'])-1})[/bold red]"
            )
            raise typer.Exit(1)

        page = data["pages"][index]
        
        # Check if raw HTML exists
        if "rawHtml" not in page:
            console.print(f"[bold red]No raw HTML found for document at index {index}[/bold red]")
            raise typer.Exit(1)
            
        raw_html = page["rawHtml"]
        title = page.get("title", f"Untitled_Page_{index}")
        
        # Display document info
        console.print(
            Panel(
                f"[bold]Document Information[/bold]\n\n"
                f"Title: [cyan]{title}[/cyan]\n"
                f"Index: {index}\n"
                f"URL: {page.get('url', 'N/A')}\n",
                title="Raw HTML Source",
                border_style="green",
            )
        )
        
        # Display raw HTML
        console.print("\n[bold]Raw HTML:[/bold]")
        console.print(Syntax(raw_html, "html", theme="monokai", line_numbers=True))
        
    except Exception as e:
        console.print(f"[bold red]Error loading document: {str(e)}[/bold red]")
        raise typer.Exit(1)


def register_commands(app: typer.Typer):
    """Register ingest commands with the main app."""
    app.add_typer(
        ingest_app, name="ingest", help="Document ingestion pipeline commands"
    )

