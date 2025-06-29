"""Feature engineering pipeline for document processing."""

from typing import Optional

from zenml import pipeline

# Import and register custom materializers
from ..zenml_setup import register_materializers

register_materializers()

from help_rag.steps.convert_to_markdown import convert_to_markdown
from help_rag.steps.chunk_document import chunk_document
from help_rag.steps.add_document_context import add_document_context
from help_rag.steps.add_image_descriptions import add_image_descriptions
from help_rag.steps.load_to_document_store import load_to_document_store
from help_rag.steps.load_to_vector_db import load_to_vector_db


@pipeline(enable_cache=True)
def feature_engineering_pipeline(
    index: Optional[int] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    all_docs: bool = True,
    source_path: str = "output/epic-docs.json",
    images_dir: str = "output/images",
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
    dynamic_chunking: bool = True,
    skip_enrichment: bool = False,
    skip_image_descriptions: bool = False,
    dry_run: bool = False,
):
    """Run the feature engineering pipeline for document processing.

    This pipeline follows these steps:
    1. Convert HTML to Markdown
    2. Chunk documents
    3. Add document context
    4. Add image descriptions
    5. Load to document store
    6. Embed and load to vector database

    Args:
        index: Optional specific index of the document to process
        offset: Starting index for batch processing
        limit: Maximum number of documents to process
        all_docs: Whether to process all documents
        source_path: Path to the source JSON file
        images_dir: Directory where images are stored
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        dynamic_chunking: Whether to use dynamic chunking
        skip_enrichment: Whether to skip contextual enrichment
        skip_image_descriptions: Whether to skip adding image descriptions
        dry_run: Whether to skip saving to the databases
    """
    # Convert HTML to Markdown
    documents = convert_to_markdown(
        index=index,
        offset=offset,
        limit=limit,
        all_docs=all_docs,
        source_path=source_path,
        images_dir=images_dir,
    )

    # Chunk documents
    doc_chunks = chunk_document(
        documents=documents,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        dynamic_chunking=dynamic_chunking,
    )

    # Add document context
    enriched_chunks = add_document_context(
        doc_chunks=doc_chunks, skip_enrichment=skip_enrichment
    )

    # Add image descriptions
    with_images = add_image_descriptions(
        doc_chunks=enriched_chunks, skip_image_descriptions=skip_image_descriptions
    )

    # Load to document store
    stored_docs = load_to_document_store(doc_chunks=with_images, dry_run=dry_run)

    # Embed and load to vector database
    load_to_vector_db(doc_chunks=stored_docs, dry_run=dry_run)
