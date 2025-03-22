"""Command modules for the CLI interface."""

# Re-export named functions with explicit names for better imports
from .document_commands import register_commands as register_document_commands
from .query_commands import register_commands as register_query_commands
from .db_commands import register_commands as register_db_commands
from .evaluation_commands import register_commands as register_evaluation_commands
from .utility_commands import register_commands as register_utility_commands
from .pipeline_commands import register_commands as register_pipeline_commands
from .ingest_commands import register_commands as register_ingest_commands

__all__ = [
    "register_document_commands",
    "register_query_commands",
    "register_db_commands",
    "register_evaluation_commands",
    "register_utility_commands",
    "register_pipeline_commands",
    "register_ingest_commands",
]
