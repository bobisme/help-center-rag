"""Shared utilities and common functionality for CLI commands."""

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.markdown import Markdown
from loguru import logger

# Rich console for pretty output
console = Console()

# Configure loguru
logger.configure(
    handlers=[
        {
            "sink": lambda msg: console.print(
                f"[dim]{msg}[/dim]", markup=True, highlight=False
            )
        }
    ]
)


def create_progress_bar(description: str = "Processing"):
    """Create a rich progress bar."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn()
    )


def display_table(title: str, columns: list, rows: list):
    """Display data in a formatted table."""
    table = Table(title=title)

    for column in columns:
        table.add_column(column)

    for row in rows:
        table.add_row(*row)

    console.print(table)


def display_document_info(doc):
    """Display document information in a formatted panel."""
    metadata_str = "\n".join([f"{k}: {v}" for k, v in doc.metadata.items()])

    panel_content = f"""
# {doc.title}

**ID**: {doc.id}
**Path**: {doc.source_path}
**Created**: {doc.created_at}
**Word Count**: {doc.word_count}
**Chunk Count**: {len(doc.chunks) if hasattr(doc, 'chunks') else 'N/A'}

## Metadata
{metadata_str}
"""

    console.print(Panel(Markdown(panel_content), title="Document Information"))
