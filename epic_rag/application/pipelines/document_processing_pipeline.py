"""ZenML pipeline for processing documents from Epic documentation."""

from typing import List, Dict, Any, Optional

from zenml import pipeline, step
from zenml.config import DockerSettings

# Output typing for steps

from ...domain.models.document import Document
from ...infrastructure.container import container
from ..use_cases.ingest_document import IngestDocumentUseCase


@step
def load_markdown_documents(
    source_dir: str, file_pattern: str = "*.md", limit: Optional[int] = None
) -> List[Document]:
    """Load markdown documents from the source directory.

    Args:
        source_dir: Directory containing markdown files
        file_pattern: Pattern to match markdown files
        limit: Optional limit on number of files to process

    Returns:
        List of document objects
    """
    import glob
    import os
    from pathlib import Path

    # Find all markdown files matching the pattern
    file_paths = glob.glob(os.path.join(source_dir, file_pattern))

    # Sort by modification time (newest first)
    file_paths = sorted(file_paths, key=os.path.getmtime, reverse=True)

    # Apply limit if specified
    if limit:
        file_paths = file_paths[:limit]

    documents = []
    for file_path in file_paths:
        # Read the file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract title from the filename or first heading
        filename = os.path.basename(file_path)
        title = Path(filename).stem.replace("_", " ").title()

        # Create the document
        document = Document(
            title=title,
            content=content,
            metadata={
                "source_path": file_path,
                "file_type": "markdown",
                "filename": filename,
            },
        )
        documents.append(document)

    print(f"Loaded {len(documents)} markdown documents from {source_dir}")
    return documents


@step
def preprocess_documents(documents: List[Document]) -> List[Document]:
    """Preprocess documents for better chunking and retrieval.

    Args:
        documents: List of documents to preprocess

    Returns:
        List of preprocessed documents
    """
    preprocessed_documents = []

    for document in documents:
        # Process title to add as heading if not present
        if not document.content.startswith("# "):
            document.content = f"# {document.title}\n\n{document.content}"

        # Add to processed list
        preprocessed_documents.append(document)

    return preprocessed_documents


@step
def ingest_documents(
    documents: List[Document],
    dynamic_chunking: bool = True,
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """Ingest documents into the system.

    Args:
        documents: List of documents to ingest
        dynamic_chunking: Whether to use dynamic chunking
        min_chunk_size: Minimum chunk size when using dynamic chunking
        max_chunk_size: Maximum chunk size when using dynamic chunking
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with ingestion statistics
    """
    import asyncio
    from ...infrastructure.container import container

    # Get required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    chunking_service = container.get("chunking_service")
    embedding_service = container.get("embedding_service")

    # Create use case
    use_case = IngestDocumentUseCase(
        document_repository=document_repository,
        vector_repository=vector_repository,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )

    # Process all documents
    async def process_all():
        results = []
        for document in documents:
            result = await use_case.execute(
                document=document,
                dynamic_chunking=dynamic_chunking,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.append(result)
        return results

    processed_documents = asyncio.run(process_all())

    # Collect statistics
    total_chunks = sum(doc.chunk_count for doc in processed_documents)

    return {
        "processed_document_count": len(processed_documents),
        "total_chunk_count": total_chunks,
        "avg_chunks_per_document": (
            total_chunks / len(processed_documents) if processed_documents else 0
        ),
    }


@pipeline(enable_cache=True)
def document_processing_pipeline(
    source_dir: str,
    file_pattern: str = "*.md",
    limit: Optional[int] = None,
    dynamic_chunking: bool = True,
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """Pipeline for processing and ingesting documents.

    Args:
        source_dir: Directory containing markdown files
        file_pattern: Pattern to match markdown files
        limit: Optional limit on number of files to process
        dynamic_chunking: Whether to use dynamic chunking
        min_chunk_size: Minimum chunk size when using dynamic chunking
        max_chunk_size: Maximum chunk size when using dynamic chunking
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with processing statistics
    """
    # Step 1: Load documents
    documents = load_markdown_documents(
        source_dir=source_dir, file_pattern=file_pattern, limit=limit
    )

    # Step 2: Preprocess documents
    preprocessed_documents = preprocess_documents(documents=documents)

    # Step 3: Ingest documents
    stats = ingest_documents(
        documents=preprocessed_documents,
        dynamic_chunking=dynamic_chunking,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return stats
