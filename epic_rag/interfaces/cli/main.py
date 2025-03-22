"""Command-line interface for Epic Documentation RAG system."""

import typer

from ...infrastructure.container import setup_container
from .commands import (
    register_document_commands,
    register_query_commands,
    register_db_commands,
    register_evaluation_commands,
    register_utility_commands,
    register_pipeline_commands,
)
from .commands.vis import register_vis_commands
from .commands.images import register_image_commands
from .commands.testing import register_testing_commands
from .commands.help_center import register_help_center_commands
from .commands.ingest_commands import register_commands as register_ingest_commands

# Create Typer app
app = typer.Typer(
    name="epic-rag",
    help="Epic Documentation RAG System",
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.callback()
def callback():
    """Epic Documentation RAG System.

    A retrieval-augmented generation system for Epic documentation
    using Anthropic's Contextual Retrieval methodology.
    """
    # Initialize the service container
    setup_container()


# Import command functions for backward compatibility aliases - these imports
# need to be here rather than at the top to avoid circular imports
# flake8: noqa: E402
from .commands.document_commands import ingest_documents
from .commands.query_commands import query, bm25_search, hybrid_search, transform_query
from .commands.evaluation_commands import (
    test_enrichment,
    benchmark_bm25,
    evaluate_enrichment,
)
from .commands.utility_commands import show_info
from .commands.pipeline_commands import run_zenml_pipeline
from .commands.vis.document_commands import show_chunks, show_document_by_id
from .commands.images.image_commands import (
    describe_images,
    describe_images_smolvlm,
    compare_descriptions,
)
from .commands.testing.testing_commands import (
    demo_enrichment,
    evaluate_enrichment as eval_enrichment,
    test_help_center_pipeline,
)
from .commands.help_center.help_center_commands import (
    process_help_center,
    list_help_center_docs,
    run_help_center_pipeline,
)
from .commands.ingest_commands import ingest_app as new_ingest_app

# Register top-level alias commands for backward compatibility with Justfile
app.command("old-ingest")(ingest_documents)

# Replace the ingest command with the new ingest app
app.add_typer(new_ingest_app, name="ingest")
app.command("query")(query)
app.command("bm25")(bm25_search)
app.command("hybrid-search")(hybrid_search)
app.command("transform-query")(transform_query)
app.command("test-enrichment")(test_enrichment)
app.command("benchmark-bm25")(benchmark_bm25)
app.command("evaluate-enrichment")(evaluate_enrichment)
app.command("info")(show_info)
app.command("zenml-run")(run_zenml_pipeline)

# Register new aliases for visualization commands
app.command("show-doc-chunks")(show_chunks)
app.command("show-doc-by-id")(show_document_by_id)

# Register new aliases for image commands
app.command("image-describe")(describe_images)
app.command("smolvlm-describe")(describe_images_smolvlm)
app.command("compare-descriptions")(compare_descriptions)

# Register new aliases for testing commands
app.command("enrichment-demo")(demo_enrichment)
app.command("manual-evaluation")(eval_enrichment)
app.command("test-help-center")(test_help_center_pipeline)

# Register new aliases for help center commands
app.command("process-help-center")(process_help_center)
app.command("list-help-center")(list_help_center_docs)
app.command("run-help-center")(run_help_center_pipeline)


def main():
    """Run the CLI application."""
    # Register all command modules
    register_document_commands(app)
    register_query_commands(app)
    register_db_commands(app)
    register_evaluation_commands(app)
    register_utility_commands(app)
    register_pipeline_commands(app)

    # Register new command modules
    register_vis_commands(app)
    register_image_commands(app)
    register_testing_commands(app)
    register_help_center_commands(app)
    register_ingest_commands(app)

    # Run the Typer app
    app()


if __name__ == "__main__":
    main()
