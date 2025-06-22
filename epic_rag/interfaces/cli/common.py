"""Common utilities for CLI commands."""

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

# Create a shared console instance
console = Console()


def create_progress_bar(description: str = "Processing"):
    """Create a rich progress bar."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[status]}"),
    )
