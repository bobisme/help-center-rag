"""Commands for running ZenML pipelines."""

import typer
from rich.panel import Panel

from ....pipelines.feature_engineering import feature_engineering_pipeline
from .common import console

pipeline_app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Commands for running ZenML pipelines",
)


def run_zenml_pipeline(
    pipeline_name: str = typer.Option(..., help="Name of the pipeline to run"), **kwargs
):
    """Run a ZenML pipeline by name.

    This is a legacy function for backward compatibility.
    """
    console.print(
        f"[yellow]Warning: Use 'rag pipeline {pipeline_name}' instead[/yellow]"
    )

    if pipeline_name == "feature-engineering":
        run_feature_engineering(**kwargs)
    else:
        console.print(f"[bold red]Unknown pipeline: {pipeline_name}[/bold red]")
        raise typer.Exit(1)


@pipeline_app.command("feature-engineering")
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
    """Run the feature engineering pipeline for document processing.

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
            f"[bold]Running Feature Engineering Pipeline{mode_text}[/bold]\n\n"
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
        "[bold green]Feature engineering pipeline completed successfully![/bold green]"
    )


def register_commands(app: typer.Typer):
    """Register pipeline commands with the main app."""
    app.add_typer(pipeline_app, name="pipeline", help="Run ZenML pipelines")
