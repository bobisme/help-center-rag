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

# Create Typer app
app = typer.Typer(
    name="epic-rag",
    help="Epic Documentation RAG System",
    add_completion=False,
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

# Register top-level alias commands for backward compatibility with Justfile
app.command("ingest")(ingest_documents)
app.command("query")(query)
app.command("bm25")(bm25_search)
app.command("hybrid-search")(hybrid_search)
app.command("transform-query")(transform_query)
app.command("test-enrichment")(test_enrichment)
app.command("benchmark-bm25")(benchmark_bm25)
app.command("evaluate-enrichment")(evaluate_enrichment)
app.command("info")(show_info)
app.command("zenml-run")(run_zenml_pipeline)


def main():
    """Run the CLI application."""
    # Register all command modules
    register_document_commands(app)
    register_query_commands(app)
    register_db_commands(app)
    register_evaluation_commands(app)
    register_utility_commands(app)
    register_pipeline_commands(app)

    # Run the Typer app
    app()


if __name__ == "__main__":
    main()
