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
from .commands.ingest_commands import register_commands as register_ingest_commands
from .commands.answer_commands import register_commands as register_answer_commands

# something
from .commands.query_commands import query
from .commands.utility_commands import show_info
from .commands.pipeline_commands import run_feature_engineering
from .commands.ingest_commands import ingest_app
from .commands.answer_commands import ask

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


# Replace the ingest command with the new ingest app
app.add_typer(ingest_app, name="ingest")
app.command("query")(query)
app.command("info")(show_info)
app.command("pipeline-feature-engineering")(run_feature_engineering)
app.command("ask")(ask)


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
    register_ingest_commands(app)
    register_answer_commands(app)

    # Run the Typer app
    app()


if __name__ == "__main__":
    main()
