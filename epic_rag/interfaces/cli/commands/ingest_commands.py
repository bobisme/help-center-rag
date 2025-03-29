"""Commands for the ingestion pipeline."""

import asyncio
import json
import os
import typer
from rich.panel import Panel
from rich.syntax import Syntax

from ....domain.models.document import Document, DocumentChunk
from ....infrastructure.container import container, setup_container
from ....application.use_cases.ingest_document import IngestDocumentUseCase
from ....pipelines.feature_engineering import feature_engineering_pipeline
from .common import console, create_progress_bar

ingest_app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Document ingestion pipeline commands for inspecting the processing stages",
)


def load_documents_from_json(
    index: int = None, offset: int = 0, limit: int = None, all_docs: bool = False
) -> list:
    """Load documents from the epic-docs.json file.

    Args:
        index: Optional specific index of the document to load
        offset: Starting index for batch loading
        limit: Maximum number of documents to load
        all_docs: Whether to load all documents

    Returns:
        A list of Document objects
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

        total_pages = len(data["pages"])
        console.print(f"Found [bold]{total_pages}[/bold] pages in {json_path}")

        # Determine which pages to process
        if index is not None:
            if index >= total_pages:
                console.print(
                    f"[bold red]Index out of range: {index} (max: {total_pages-1})[/bold red]"
                )
                raise typer.Exit(1)
            pages_to_process = [index]
        elif all_docs:
            pages_to_process = range(total_pages)
        else:
            # Apply offset and limit
            start_idx = min(offset, total_pages)
            if limit is not None:
                end_idx = min(start_idx + limit, total_pages)
            else:
                end_idx = total_pages
            pages_to_process = range(start_idx, end_idx)

        documents = []
        images_dir_checked = False
        images_dir = "output/images"

        for idx in pages_to_process:
            page = data["pages"][idx]

            # Extract document data
            title = page.get("title", f"Untitled_Page_{idx}")
            page_id = idx  # Use index as ID
            category = page.get("metadata", {}).get("path", ["Uncategorized"])[0]
            updated_at = page.get("metadata", {}).get("crawlDate")

            # Convert HTML content to markdown (assuming it has already been converted)
            content = page.get("content", "")
            if not content and "rawHtml" in page:
                from html2md import convert_html_to_markdown, preprocess_html

                # Only check images directory once
                if not images_dir_checked:
                    if os.path.exists(images_dir):
                        console.print(f"[green]Using images from {images_dir}[/green]")
                    else:
                        console.print(
                            f"[yellow]Warning: Images directory {images_dir} not found. Images will be removed.[/yellow]"
                        )
                    images_dir_checked = True

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
                    "page_index": idx,
                },
            )

            documents.append(document)

        return documents

    except Exception as e:
        console.print(f"[bold red]Error loading documents: {str(e)}[/bold red]")
        raise typer.Exit(1)


def load_document_from_json(index: int) -> Document:
    """Load a single document from the epic-docs.json file by index.

    Args:
        index: The index of the document to load

    Returns:
        The loaded document
    """
    documents = load_documents_from_json(index=index)
    return documents[0]


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
        None, "--index", "-i", help="Document index in epic-docs.json"
    ),
    limit: int = typer.Option(
        None, "--limit", "-l", help="Limit number of documents to process"
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset to start processing from"
    ),
    all_docs: bool = typer.Option(
        False, "--all", help="Process all documents in the file"
    ),
    with_context: bool = typer.Option(
        False, "--with-context", help="Show chunks with enriched context"
    ),
    with_image_descriptions: bool = typer.Option(
        False, "--with-image-descriptions", help="Show chunks with image descriptions"
    ),
):
    """Show the chunks for documents from epic-docs.json."""
    # Validate parameters
    if index is None and not (limit or all_docs or offset > 0):
        console.print(
            "[bold red]Error: Must specify either --index, --limit, --offset, or --all[/bold red]"
        )
        raise typer.Exit(1)

    # Load documents
    documents = load_documents_from_json(
        index=index, offset=offset, limit=limit, all_docs=all_docs
    )

    if not documents:
        console.print("[bold red]No documents found to process[/bold red]")
        raise typer.Exit(1)

    # Initialize container
    setup_container()

    # Get services
    chunking_service = container.get("chunking_service")
    contextual_enrichment_service = container.get("contextual_enrichment_service")
    image_description_service = container.get("image_description_service")

    # Use a single async function to process all documents
    import re

    async def process_all_documents(progress_task):
        all_chunks = []
        total_img_count = 0

        for doc_index, doc in enumerate(documents):
            try:
                # Update progress display
                progress.update(
                    progress_task,
                    advance=0,
                    description=f"Processing {doc_index+1}/{len(documents)}",
                    status=f"{doc.title}",
                )

                # Chunk the document
                chunks = await chunking_service.dynamic_chunk_document(
                    document=doc,
                    min_chunk_size=300,
                    max_chunk_size=800,
                )

                # Count images in all chunks
                doc_total_images = 0
                for chunk in chunks:
                    image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
                    matches = list(re.finditer(image_pattern, chunk.content))
                    doc_total_images += len(matches)

                total_img_count += doc_total_images

                # Handle contextual enrichment and image descriptions based on flags
                if with_context and contextual_enrichment_service:
                    # Special handling when both context and image descriptions are requested
                    if with_image_descriptions and hasattr(
                        contextual_enrichment_service, "_image_description_service"
                    ):
                        # Get descriptions first to populate cache
                        chunks = await contextual_enrichment_service.enrich_chunks(
                            document=doc, chunks=chunks
                        )

                        # Now get image descriptions but add them under the images instead of at the top
                        if hasattr(image_description_service, "process_chunk_images"):
                            for i, chunk in enumerate(chunks):
                                # Get just the context (not image descriptions)
                                context = chunk.metadata.get("context", "")

                                # Create new chunk with just context at top
                                content_parts = chunk.content.split("\n\n")
                                # Skip the context and the image descriptions at the top
                                original_content = (
                                    "\n\n".join(content_parts[2:])
                                    if len(content_parts) > 2
                                    else chunk.content
                                )
                                clean_chunk = DocumentChunk(
                                    id=chunk.id,
                                    document_id=chunk.document_id,
                                    content=f"{context}\n\n{original_content}",
                                    metadata=chunk.metadata,
                                    embedding=chunk.embedding,
                                    chunk_index=chunk.chunk_index,
                                    previous_chunk_id=getattr(
                                        chunk, "previous_chunk_id", None
                                    ),
                                    next_chunk_id=getattr(chunk, "next_chunk_id", None),
                                    relevance_score=getattr(
                                        chunk, "relevance_score", None
                                    ),
                                )

                                # Process to add image descriptions under images
                                processed_chunk = await image_description_service.process_chunk_images(
                                    clean_chunk
                                )
                                chunks[i] = processed_chunk
                    else:
                        # Just apply contextual enrichment without fixing image descriptions
                        chunks = await contextual_enrichment_service.enrich_chunks(
                            document=doc, chunks=chunks
                        )

                # Apply only image descriptions if requested without context
                elif with_image_descriptions and hasattr(
                    image_description_service, "process_chunk_images"
                ):
                    # Process all chunks with image descriptions
                    for i, chunk in enumerate(chunks):
                        updated_chunk = (
                            await image_description_service.process_chunk_images(chunk)
                        )
                        chunks[i] = updated_chunk  # Update the chunk in the list

                # Add document title to chunks for grouping
                for chunk in chunks:
                    chunk.document_title = doc.title

                all_chunks.extend(chunks)

                # Update progress
                progress.update(progress_task, advance=1)

            except Exception as e:
                console.print(
                    f"[bold red]Error processing document {doc.title}: {str(e)}[/bold red]"
                )
                progress.update(progress_task, advance=1)

        return all_chunks, total_img_count

    # Create a progress bar and execute all processing in one event loop
    with create_progress_bar() as progress:
        task = progress.add_task(
            "Processing documents", total=len(documents), status=""
        )

        # Execute all document processing in a single event loop
        all_document_chunks, total_images = asyncio.run(process_all_documents(task))

    # Display image description summary if applicable
    if with_image_descriptions:
        if total_images > 0:
            console.print(
                f"[green]Added descriptions for {total_images} images across {len(all_document_chunks)} chunks[/green]"
            )
        else:
            console.print("[yellow]No images found in documents[/yellow]")

    # Display chunks
    console.print(f"\n[bold]Chunks ([cyan]{len(all_document_chunks)}[/cyan]):[/bold]")

    # Group chunks by document for better organization
    from itertools import groupby

    # Sort chunks by document title for grouping
    all_document_chunks.sort(key=lambda x: getattr(x, "document_title", ""))

    # Group by document title
    for doc_title, doc_chunks in groupby(
        all_document_chunks, key=lambda x: getattr(x, "document_title", "")
    ):
        doc_chunks = list(doc_chunks)
        console.print(f"\n[bold green]Document: {doc_title}[/bold green]")
        console.print(f"Number of chunks: {len(doc_chunks)}")

        for i, chunk in enumerate(doc_chunks):
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
            # Display content with wrapping enabled for better readability of image descriptions
            console.print(
                Syntax(chunk.content, "markdown", theme="monokai", word_wrap=True)
            )

            # If the chunk has a context field from enrichment, show it
            if with_context and hasattr(chunk, "context") and chunk.context:
                console.print("\n[bold yellow]Context:[/bold yellow]")
                console.print(chunk.context)


@ingest_app.command("embed")
def embed_document(
    index: int = typer.Option(
        None, "--index", "-i", help="Document index in epic-docs.json"
    ),
    limit: int = typer.Option(
        None, "--limit", "-l", help="Limit number of documents to process"
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset to start processing from"
    ),
    all_docs: bool = typer.Option(
        False, "--all", help="Process all documents in the file"
    ),
):
    """Embed chunks for documents from epic-docs.json and print vector information."""
    # Validate parameters
    if index is None and not (limit or all_docs or offset > 0):
        console.print(
            "[bold red]Error: Must specify either --index, --limit, --offset, or --all[/bold red]"
        )
        raise typer.Exit(1)

    # Load documents
    documents = load_documents_from_json(
        index=index, offset=offset, limit=limit, all_docs=all_docs
    )

    if not documents:
        console.print("[bold red]No documents found to process[/bold red]")
        raise typer.Exit(1)

    # Initialize container
    setup_container()

    # Get services
    chunking_service = container.get("chunking_service")
    contextual_enrichment_service = container.get("contextual_enrichment_service")
    embedding_service = container.get("embedding_service")
    image_description_service = container.get("image_description_service")

    # Create a single async function to process all documents
    import re

    async def process_all_documents(progress_task):
        all_embedded = []
        total_img_count = 0

        for doc_index, doc in enumerate(documents):
            try:
                # Update progress display
                progress.update(
                    progress_task,
                    advance=0,
                    description=f"Processing {doc_index+1}/{len(documents)}",
                    status=f"{doc.title}",
                )

                # Chunk the document
                chunks = await chunking_service.dynamic_chunk_document(
                    document=doc,
                    min_chunk_size=300,
                    max_chunk_size=800,
                )

                # Apply contextual enrichment
                chunks = await contextual_enrichment_service.enrich_chunks(
                    document=doc, chunks=chunks
                )

                # Add image descriptions
                doc_total_images = 0
                if hasattr(image_description_service, "process_chunk_images"):
                    # Process all chunks with image descriptions
                    for i, chunk in enumerate(chunks):
                        # Count images in the chunk
                        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
                        matches = list(re.finditer(image_pattern, chunk.content))
                        doc_total_images += len(matches)

                        # Process image descriptions
                        chunks[i] = (
                            await image_description_service.process_chunk_images(chunk)
                        )

                total_img_count += doc_total_images

                # Generate embeddings
                embedded_chunks = await embedding_service.batch_embed_chunks(chunks)

                # Add document title for display grouping
                for chunk in embedded_chunks:
                    chunk.document_title = doc.title

                all_embedded.extend(embedded_chunks)

                # Update progress
                progress.update(progress_task, advance=1)

            except Exception as e:
                console.print(
                    f"[bold red]Error processing document {doc.title}: {str(e)}[/bold red]"
                )
                progress.update(progress_task, advance=1)

        return all_embedded, total_img_count

    # Create a progress bar and execute all processing in one event loop
    with create_progress_bar() as progress:
        task = progress.add_task(
            "Processing documents", total=len(documents), status=""
        )

        # Execute all document processing in a single event loop
        all_embedded_chunks, total_images = asyncio.run(process_all_documents(task))

    if not all_embedded_chunks:
        console.print("[bold red]No chunks were successfully embedded[/bold red]")
        raise typer.Exit(1)

    # Report image description summary if applicable
    if total_images > 0:
        console.print(
            f"[green]Added descriptions for {total_images} images across {len(all_embedded_chunks)} chunks[/green]"
        )

    # Display results summary
    console.print(
        f"\n[bold green]Successfully embedded {len(all_embedded_chunks)} chunks from {len(documents)} documents[/bold green]"
    )

    # Group chunks by document for better organization
    from itertools import groupby

    # Sort chunks by document title for grouping
    all_embedded_chunks.sort(key=lambda x: getattr(x, "document_title", ""))

    # Group by document title
    for doc_title, doc_chunks in groupby(
        all_embedded_chunks, key=lambda x: getattr(x, "document_title", "")
    ):
        doc_chunks = list(doc_chunks)
        console.print(f"\n[bold green]Document: {doc_title}[/bold green]")
        console.print(f"Number of embedded chunks: {len(doc_chunks)}")

        # Display embedding info for first chunk only if multiple documents
        if len(documents) > 1:
            chunk = doc_chunks[0]
            console.print(f"\n[bold cyan]First Chunk Preview[/bold cyan]")
            console.print(f"Content Preview: {chunk.content[:50]}...")

            if hasattr(chunk, "embedding") and chunk.embedding:
                vector = chunk.embedding
                console.print(f"Embedding Vector (dim={len(vector)})")
                console.print(f"First 5 values: {vector[:5]}")
        else:
            # For single document, display all chunks
            for i, chunk in enumerate(doc_chunks):
                console.print(f"\n[bold cyan]Chunk {i+1}[/bold cyan]")
                console.print(f"Content Preview: {chunk.content[:50]}...")

                if hasattr(chunk, "embedding") and chunk.embedding:
                    vector = chunk.embedding
                    console.print(f"Embedding Vector (dim={len(vector)})")
                    console.print(f"First 5 values: {vector[:5]}")
                    console.print(f"Last 5 values: {vector[-5:]}")


@ingest_app.command("pipeline")
def run_pipeline(
    index: int = typer.Option(
        None, "--index", "-i", help="Document index in epic-docs.json"
    ),
    limit: int = typer.Option(
        None, "--limit", "-l", help="Limit number of documents to process"
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset to start processing from"
    ),
    all_docs: bool = typer.Option(
        False, "--all", help="Process all documents in the file"
    ),
    no_enrich: bool = typer.Option(
        False, "--no-enrich", help="Skip contextual enrichment"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Process without saving to database"
    ),
    show_chunks: bool = typer.Option(
        False,
        "--show-chunks",
        help="Show all processed chunks (recommended for dry runs only)",
    ),
):
    """Run the full ingestion pipeline for documents from epic-docs.json.

    Use --index for a single document, or --limit/--offset/--all for batch processing.
    """
    # Validate parameters
    if index is None and not (limit or all_docs or offset > 0):
        console.print(
            "[bold red]Error: Must specify either --index, --limit, --offset, or --all[/bold red]"
        )
        raise typer.Exit(1)

    # Load documents
    documents = load_documents_from_json(
        index=index, offset=offset, limit=limit, all_docs=all_docs
    )

    if not documents:
        console.print("[bold red]No documents found to process[/bold red]")
        raise typer.Exit(1)

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
            f"Documents to process: [cyan]{len(documents)}[/cyan]\n"
            f"{'Index: ' + str(index) if index is not None else 'Batch processing'}\n"
            f"{'Offset: ' + str(offset) if offset > 0 else ''}\n"
            f"{'Limit: ' + str(limit) if limit is not None else ''}\n"
            f"{'Processing all documents' if all_docs else ''}\n",
            title="Pipeline Execution",
            border_style="green",
        )
    )

    # Create a single async function to process all documents
    async def process_all_documents(progress_task):
        results = []
        failures = []

        for doc_index, doc in enumerate(documents):
            try:
                # Update progress display
                progress.update(
                    progress_task,
                    advance=0,
                    description=f"Processing {doc_index+1}/{len(documents)}",
                    status=f"{doc.title}",
                )

                # Execute the use case for this document
                result = await ingest_use_case.execute(
                    document=doc,
                    dynamic_chunking=True,
                    min_chunk_size=300,
                    max_chunk_size=800,
                    chunk_overlap=50,
                    apply_enrichment=not no_enrich,
                    dry_run=dry_run,
                )
                results.append(result)

                # Update progress
                progress.update(progress_task, advance=1)

            except Exception as e:
                # Log the error but continue with other documents
                console.print(
                    f"[bold red]Error processing document {doc.title}: {str(e)}[/bold red]"
                )
                failures.append((doc.title, str(e)))
                progress.update(progress_task, advance=1)

        return results, failures

    # Create a progress bar and run the async process with a single event loop
    with create_progress_bar() as progress:
        task = progress.add_task(
            "Processing documents", total=len(documents), status=""
        )

        # Execute all document processing in a single event loop
        all_results, failed_docs = asyncio.run(process_all_documents(task))

    # Display summary
    console.print(f"\n[bold green]Processing completed![/bold green]")
    console.print(f"Documents processed: {len(all_results)}/{len(documents)}")

    if failed_docs:
        console.print(f"[bold red]Failed documents: {len(failed_docs)}[/bold red]")
        for title, error in failed_docs:
            console.print(f"  - {title}: {error}")

    # Display chunk counts per document
    total_chunks = sum(len(result.chunks) for result in all_results)
    console.print(f"Total chunks created: {total_chunks}")

    if dry_run:
        console.print(
            "[yellow]Dry run completed - no data was saved to the database.[/yellow]"
        )
    else:
        console.print("Documents and vectors stored in the database.")

    # Optionally show detailed chunk information
    if show_chunks:
        console.print(
            "\n[bold]Displaying processed chunks with enrichment and image descriptions:[/bold]"
        )

        # Only show chunks for the first document if processing multiple
        if len(all_results) > 1 and not dry_run:
            console.print(
                "[yellow]Showing chunks for the first document only. Use --dry-run to see all chunks.[/yellow]"
            )
            docs_to_show = [all_results[0]]
        else:
            docs_to_show = all_results

        for doc_idx, result in enumerate(docs_to_show):
            console.print(
                f"\n[bold cyan]Document {doc_idx+1}: {result.title}[/bold cyan]"
            )
            console.print(f"ID: {result.id}")
            console.print(f"Chunks: {len(result.chunks)}")

            for i, chunk in enumerate(result.chunks):
                console.print(f"\n[bold blue]Chunk {i+1}[/bold blue]")
                console.print(f"ID: {chunk.id}")
                console.print(f"Chunk Index: {chunk.chunk_index}")

                if hasattr(chunk, "token_count") and chunk.token_count:
                    console.print(f"Token Count: {chunk.token_count}")

                if chunk.metadata:
                    console.print("\n[bold]Metadata:[/bold]")
                    for key, value in chunk.metadata.items():
                        console.print(f"  {key}: {value}")

                # Display content with wrapping enabled for better readability
                console.print("\n[bold]Content:[/bold]")
                console.print(
                    Syntax(chunk.content, "markdown", theme="monokai", word_wrap=True)
                )

                # If the chunk has a context field from enrichment, show it
                if hasattr(chunk, "context") and chunk.context:
                    console.print("\n[bold yellow]Context:[/bold yellow]")
                    console.print(chunk.context)


@ingest_app.command("show-raw")
def show_raw_html(
    index: int = typer.Option(
        None, "--index", "-i", help="Document index in epic-docs.json"
    ),
    limit: int = typer.Option(
        1, "--limit", "-l", help="Limit number of documents to process"
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset to start processing from"
    ),
    all_docs: bool = typer.Option(
        False, "--all", help="Show raw HTML for all documents (not recommended)"
    ),
):
    """Show the raw HTML content for documents from epic-docs.json."""
    # Validate parameters
    if index is None and not (limit or all_docs or offset > 0):
        console.print(
            "[bold red]Error: Must specify either --index, --limit, --offset, or --all[/bold red]"
        )
        raise typer.Exit(1)

    # Warning for large requests
    if all_docs:
        console.print(
            "[bold yellow]Warning: Displaying raw HTML for all documents may be overwhelming.[/bold yellow]"
        )
        if not typer.confirm("Continue anyway?"):
            raise typer.Exit(0)

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

        total_pages = len(data["pages"])

        # Determine which pages to process
        if index is not None:
            if index >= total_pages:
                console.print(
                    f"[bold red]Index out of range: {index} (max: {total_pages-1})[/bold red]"
                )
                raise typer.Exit(1)
            pages_to_process = [index]
        elif all_docs:
            pages_to_process = range(total_pages)
        else:
            # Apply offset and limit
            start_idx = min(offset, total_pages)
            if limit is not None:
                end_idx = min(start_idx + limit, total_pages)
            else:
                end_idx = total_pages
            pages_to_process = range(start_idx, end_idx)

        console.print(
            f"Displaying raw HTML for [bold]{len(pages_to_process)}[/bold] documents"
        )

        for idx in pages_to_process:
            page = data["pages"][idx]

            # Check if raw HTML exists
            if "rawHtml" not in page:
                console.print(
                    f"[bold yellow]No raw HTML found for document at index {idx} - skipping[/bold yellow]"
                )
                continue

            raw_html = page["rawHtml"]
            title = page.get("title", f"Untitled_Page_{idx}")

            # Display document info
            console.print(
                Panel(
                    f"[bold]Document Information[/bold]\n\n"
                    f"Title: [cyan]{title}[/cyan]\n"
                    f"Index: {idx}\n"
                    f"URL: {page.get('url', 'N/A')}\n",
                    title=f"Raw HTML Source ({idx+1}/{len(pages_to_process)})",
                    border_style="green",
                )
            )

            # Display raw HTML
            console.print("\n[bold]Raw HTML:[/bold]")
            console.print(Syntax(raw_html, "html", theme="monokai", line_numbers=True))

            # Add separator between documents
            if idx < pages_to_process[-1]:
                console.print("\n" + "-" * 80 + "\n")

                # For multiple documents, offer to continue or stop
                if (
                    len(pages_to_process) > 3
                    and idx > pages_to_process[0] + 1
                    and not typer.confirm("Continue to next document?")
                ):
                    console.print("[yellow]Stopped at user request[/yellow]")
                    break

    except Exception as e:
        console.print(f"[bold red]Error loading document: {str(e)}[/bold red]")
        raise typer.Exit(1)


@ingest_app.command("process-pipeline")
def run_feature_engineering(
    index: int = typer.Option(
        None, "--index", "-i", help="Document index in epic-docs.json"
    ),
    limit: int = typer.Option(
        None, "--limit", "-l", help="Limit number of documents to process"
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset to start processing from"
    ),
    all_docs: bool = typer.Option(
        None, "--all", help="Process all documents in the file"
    ),
    source_path: str = typer.Option(
        "output/epic-docs.json", "--source", help="Path to the source JSON file"
    ),
    images_dir: str = typer.Option(
        "output/images", "--images-dir", help="Directory where images are stored"
    ),
    min_chunk_size: int = typer.Option(
        300, "--min-chunk-size", help="Minimum chunk size in characters"
    ),
    max_chunk_size: int = typer.Option(
        800, "--max-chunk-size", help="Maximum chunk size in characters"
    ),
    chunk_overlap: int = typer.Option(
        50, "--chunk-overlap", help="Overlap between chunks in characters"
    ),
    no_dynamic_chunking: bool = typer.Option(
        False, "--no-dynamic-chunking", help="Disable dynamic chunking"
    ),
    no_enrich: bool = typer.Option(
        False, "--no-enrich", help="Skip contextual enrichment"
    ),
    no_images: bool = typer.Option(
        False, "--no-images", help="Skip image description generation"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Process without saving to database"
    ),
):
    """Run the complete document processing pipeline.

    This pipeline converts HTML to markdown, chunks documents, adds context,
    generates image descriptions, and stores the results in the document and vector databases.

    Use --index for a single document, or --limit/--offset/--all for batch processing.
    If no selection parameters are provided, all documents will be processed.
    """
    # Validate parameters
    if index is not None and (limit is not None or offset > 0 or all_docs):
        console.print(
            "[bold red]Cannot specify both --index and batch processing parameters (--limit, --offset, --all)[/bold red]"
        )
        raise typer.Exit(1)

    # Default to all_docs=True if no selection parameters are provided
    if index is None and limit is None and offset == 0 and all_docs is None:
        all_docs = True
    elif all_docs is None:
        all_docs = False

    # Display what we're about to do
    mode_text = ""
    if dry_run:
        mode_text = " [yellow](DRY RUN - No database changes)[/yellow]"
    if no_enrich:
        mode_text += " [yellow](Without enrichment)[/yellow]"
    if no_images:
        mode_text += " [yellow](Without image descriptions)[/yellow]"

    doc_selection = (
        "All documents"
        if all_docs
        else (
            f"Document at index {index}"
            if index is not None
            else f"Documents from index {offset} to {offset + limit - 1 if limit else 'end'}"
        )
    )

    console.print(
        Panel(
            f"[bold]Running Document Processing Pipeline{mode_text}[/bold]\n\n"
            f"Document Selection: [cyan]{doc_selection}[/cyan]\n"
            f"Source Path: {source_path}\n"
            f"Images Directory: {images_dir}\n"
            f"Chunking Parameters: min={min_chunk_size}, max={max_chunk_size}, overlap={chunk_overlap}, "
            f"dynamic={'Yes' if not no_dynamic_chunking else 'No'}\n",
            title="Pipeline Execution",
            border_style="green",
        )
    )

    # Run the pipeline
    feature_engineering_pipeline(
        index=index,
        offset=offset,
        limit=limit,
        all_docs=all_docs,
        source_path=source_path,
        images_dir=images_dir,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        dynamic_chunking=not no_dynamic_chunking,
        skip_enrichment=no_enrich,
        skip_image_descriptions=no_images,
        dry_run=dry_run,
    )

    console.print(
        "[bold green]Document processing pipeline completed successfully![/bold green]"
    )


def register_commands(app: typer.Typer):
    """Register ingest commands with the main app."""
    app.add_typer(
        ingest_app, name="ingest", help="Document ingestion pipeline commands"
    )
