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
def enrich_document_chunks(
    documents: List[Document], apply_enrichment: bool = True
) -> List[Document]:
    """Enrich document chunks with contextual information using LLM.

    Args:
        documents: Documents with chunks to enrich
        apply_enrichment: Whether to apply contextual enrichment (can be disabled for testing)

    Returns:
        Documents with enriched chunks
    """
    import asyncio
    from ...infrastructure.container import container

    if not apply_enrichment:
        print("Contextual enrichment skipped (disabled via parameter)")
        return documents

    # Get the contextual enrichment service
    enrichment_service = container.get("contextual_enrichment_service")

    # Process all documents
    async def process_all():
        enriched_documents = []

        for document in documents:
            if not document.chunks:
                # If document has no chunks yet, just add it as is
                enriched_documents.append(document)
                continue

            # Enrich all chunks in the document
            enriched_chunks = await enrichment_service.enrich_chunks(
                document=document, chunks=document.chunks
            )

            # Update document with enriched chunks
            document.chunks = enriched_chunks
            enriched_documents.append(document)

        return enriched_documents

    # Run the enrichment process
    result = asyncio.run(process_all())

    print(f"Enriched chunks for {len(result)} documents")
    return result


@step
def chunk_documents(
    documents: List[Document],
    dynamic_chunking: bool = True,
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Chunk documents into retrieval units.

    Args:
        documents: List of documents to chunk
        dynamic_chunking: Whether to use dynamic chunking
        min_chunk_size: Minimum chunk size when using dynamic chunking
        max_chunk_size: Maximum chunk size when using dynamic chunking
        chunk_overlap: Overlap between chunks

    Returns:
        Documents with chunks created
    """
    import asyncio
    from ...infrastructure.container import container

    # Get the chunking service
    chunking_service = container.get("chunking_service")

    # Chunk all documents
    async def chunk_all():
        chunked_documents = []
        for document in documents:
            # Chunk the document
            if dynamic_chunking:
                chunks = await chunking_service.dynamic_chunk_document(
                    document=document,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                )
            else:
                chunks = await chunking_service.chunk_document(
                    document=document,
                    chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            # Add chunks to document
            document.chunks = chunks
            chunked_documents.append(document)

        return chunked_documents

    processed_documents = asyncio.run(chunk_all())

    # Print statistics
    total_chunks = sum(doc.chunk_count for doc in processed_documents)
    print(f"Created {total_chunks} chunks from {len(processed_documents)} documents")

    return processed_documents


@step
def ingest_documents(documents: List[Document]) -> Dict[str, Any]:
    """Ingest documents with chunks into the database and vector store.

    Args:
        documents: List of documents to ingest (with chunks)

    Returns:
        Dictionary with ingestion statistics
    """
    import asyncio
    from ...infrastructure.container import container

    # Get required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    embedding_service = container.get("embedding_service")

    # Process all documents
    async def process_all():
        results = []
        for document in documents:
            # Step 1: Save the document to the repository
            saved_document = await document_repository.save_document(document)

            # Step 2: Save chunks to the repository
            for chunk in document.chunks:
                chunk.document_id = saved_document.id
                await document_repository.save_chunk(chunk)

            # Step 3: Generate embeddings for all chunks
            embedded_chunks = await embedding_service.batch_embed_chunks(
                document.chunks
            )

            # Step 4: Store embeddings in vector database
            vector_ids = await vector_repository.batch_store_embeddings(embedded_chunks)

            # Step 5: Update chunks with vector IDs
            for i, chunk in enumerate(embedded_chunks):
                chunk.vector_id = vector_ids[i]
                await document_repository.save_chunk(chunk)

            # Update the document with chunks
            saved_document.chunks = document.chunks
            results.append(saved_document)

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
    apply_enrichment: bool = True,
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
        apply_enrichment: Whether to apply contextual enrichment using LLM

    Returns:
        Dictionary with processing statistics
    """
    # Step 1: Load documents
    documents = load_markdown_documents(
        source_dir=source_dir, file_pattern=file_pattern, limit=limit
    )

    # Step 2: Preprocess documents
    preprocessed_documents = preprocess_documents(documents=documents)

    # Step 3: Chunk documents
    chunked_documents = chunk_documents(
        documents=preprocessed_documents,
        dynamic_chunking=dynamic_chunking,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Step 4: Enrich chunks with contextual information
    enriched_documents = enrich_document_chunks(
        documents=chunked_documents, apply_enrichment=apply_enrichment
    )

    # Step 5: Ingest documents
    stats = ingest_documents(documents=enriched_documents)

    return stats
