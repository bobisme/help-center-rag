"""Pipeline execution CLI commands."""

import asyncio
import typer
from typing import Optional

from ....infrastructure.container import container
from .common import console, logger

pipeline_app = typer.Typer(pretty_exceptions_enable=False)


@pipeline_app.command("zenml-run")
def run_zenml_pipeline(
    pipeline_name: str = typer.Argument(..., help="Name of the pipeline to run"),
    source_dir: Optional[str] = typer.Option(
        None, "--source-dir", "-s", help="Directory containing markdown files"
    ),
    pattern: Optional[str] = typer.Option(
        "*.md", "--pattern", "-p", help="Pattern to match markdown files"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of files to process"
    ),
):
    """Run a ZenML pipeline."""
    # Import the pipeline dynamically based on the name
    if pipeline_name == "document-processing":
        from ....application.pipelines.document_processing_pipeline import (
            DocumentProcessingPipeline,
        )

        pipeline_class = DocumentProcessingPipeline
    elif pipeline_name == "help-center":
        from ....application.pipelines.help_center_pipeline import HelpCenterPipeline

        pipeline_class = HelpCenterPipeline
    elif pipeline_name == "query-evaluation":
        from ....application.pipelines.query_evaluation_pipeline import (
            QueryEvaluationPipeline,
        )

        pipeline_class = QueryEvaluationPipeline
    elif pipeline_name == "orchestration":
        from ....application.pipelines.orchestration_pipeline import (
            OrchestrationPipeline,
        )

        pipeline_class = OrchestrationPipeline
    else:
        console.print(f"[bold red]Unknown pipeline: {pipeline_name}[/bold red]")
        console.print(
            "Available pipelines: document-processing, help-center, "
            "query-evaluation, orchestration"
        )
        return

    # Resolve the pipeline from the container
    pipeline = container.resolve(pipeline_class)

    # Set up the pipeline config
    config = {
        "source_dir": source_dir,
        "pattern": pattern,
        "limit": limit,
    }

    # Run the pipeline
    console.print(f"[bold]Running pipeline:[/bold] {pipeline_name}")

    try:
        # Different pipelines might have different interfaces
        if hasattr(pipeline, "run"):
            result = pipeline.run(**{k: v for k, v in config.items() if v is not None})
        elif hasattr(pipeline, "run_async"):
            result = asyncio.run(
                pipeline.run_async(**{k: v for k, v in config.items() if v is not None})
            )
        else:
            console.print(
                "[bold red]Pipeline has no run() or run_async() method[/bold red]"
            )
            return

        console.print("[bold green]Pipeline completed successfully[/bold green]")

        # Display the pipeline result if it returned something
        if result:
            console.print("[bold]Pipeline result:[/bold]")
            console.print(result)

    except Exception as e:
        console.print(f"[bold red]Error running pipeline:[/bold red] {str(e)}")
        logger.exception(f"Pipeline error: {str(e)}")


def register_commands(app: typer.Typer):
    """Register pipeline commands with the main app."""
    app.add_typer(pipeline_app, name="pipeline", help="Pipeline execution commands")
