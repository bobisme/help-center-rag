"""Command-line interface for Epic Documentation RAG system."""

import typer

from ...infrastructure.container import setup_container
from .commands.query_commands import query
from .commands.utility_commands import show_info
from .commands.pipeline_commands import run_feature_engineering
from .commands.ingest_commands import ingest_app
from .commands.answer_commands import ask
from .commands.db_commands import db_app
from .commands.document_commands import document_app

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


# Register the core commands that appear in the main help
app.add_typer(ingest_app, name="ingest")
app.command("query")(query)
app.command("info")(show_info)
app.command("pipeline-feature-engineering")(run_feature_engineering)
app.command("ask")(ask)

# Add database commands under the db subcommand
app.add_typer(db_app, name="db")

# Add document commands
app.add_typer(document_app, name="documents")


def main():
    """Run the CLI application."""
    # Run the Typer app
    app()


if __name__ == "__main__":
    main()
