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

# Add ingest commands for processing Help Center JSON data
app.add_typer(
    ingest_app, 
    name="ingest", 
    help="Process Help Center JSON data with specialized pipeline"
)

# Add document commands for managing general documents
app.add_typer(
    document_app, 
    name="documents", 
    help="Manage and view documents already in the system"
)

# Add search and question answering commands
app.command("query")(query)
app.command("ask")(ask)

# Add database management commands
app.add_typer(db_app, name="db", help="Database maintenance and inspection tools")

# Add utility commands
app.command("info")(show_info)
app.command("pipeline-feature-engineering")(run_feature_engineering)


def main():
    """Run the CLI application."""
    # Run the Typer app
    app()


if __name__ == "__main__":
    main()
