"""ZenML pipeline for processing Epic help center documentation."""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from zenml import pipeline, step
from zenml.config import DockerSettings

from ...domain.models.document import Document
from ...infrastructure.container import container
from ..use_cases.ingest_document import IngestDocumentUseCase


@step
def load_help_center_pages(
    json_path: str, start_index: int = 0, limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load help center pages from the JSON file.

    Args:
        json_path: Path to the JSON file
        start_index: Index to start processing from
        limit: Optional limit on number of pages to process

    Returns:
        Tuple containing list of pages and metadata
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            print(f"Error: Invalid JSON structure in {json_path}")
            return [], {"error": "Invalid JSON structure"}

        pages = data["pages"]
        total_page_count = len(pages)

        # Calculate how many pages to process
        if limit is None or limit <= 0 or limit > total_page_count - start_index:
            end_index = total_page_count
        else:
            end_index = start_index + limit

        # Get pages slice
        selected_pages = pages[start_index:end_index]

        metadata = {
            "total_pages": total_page_count,
            "selected_pages": len(selected_pages),
            "start_index": start_index,
            "end_index": end_index - 1,
        }

        print(
            f"Loaded {len(selected_pages)} of {total_page_count} pages from {json_path}"
        )
        return selected_pages, metadata

    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        return [], {"error": f"File {json_path} not found"}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return [], {"error": f"Invalid JSON format in {json_path}"}


@step
def convert_pages_to_markdown(
    pages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: str,
    images_dir: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Convert help center pages to markdown.

    Args:
        pages: List of pages from JSON
        metadata: Metadata from load step
        output_dir: Directory to save markdown output
        images_dir: Directory containing images

    Returns:
        Tuple containing list of converted documents and updated metadata
    """
    from html2md import convert_html_to_markdown, preprocess_html

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "markdown"), exist_ok=True)

    converted_docs = []

    for i, page in enumerate(pages):
        try:
            title = page.get("title", f"Untitled_Page_{i}")
            # Check for both "html" and "rawHtml" keys since the format may vary
            raw_html = page.get("html", "") or page.get("rawHtml", "")

            if not raw_html:
                print(f"Warning: Empty HTML content for page {i}")
                continue

            # Clean the title to create a valid filename
            clean_title = (
                title.replace(" ", "_").replace("/", "-").replace(":", "").lower()
            )

            # Convert HTML to markdown
            markdown = convert_html_to_markdown(
                raw_html, images_dir=images_dir, heading_style="ATX", wrap=True
            )

            # Save markdown to file
            markdown_path = os.path.join(output_dir, "markdown", f"{clean_title}.md")
            with open(markdown_path, "w") as f:
                f.write(markdown)

            converted_docs.append(
                {
                    "title": title,
                    "path": markdown_path,
                    "url": page.get("url", ""),
                    "original_index": str(
                        i
                    ),  # Convert to string to fix pydantic validation
                }
            )

        except Exception as e:
            print(f"Error processing page {i}: {str(e)}")

    # Update metadata
    metadata["converted_count"] = len(converted_docs)
    metadata["success_rate"] = len(converted_docs) / len(pages) if pages else 0

    print(f"Successfully converted {len(converted_docs)} of {len(pages)} pages")
    return converted_docs, metadata


@step
def generate_documents_from_markdown(
    converted_docs: List[Dict[str, str]], metadata: Dict[str, Any]
) -> List[Document]:
    """Generate Document objects from markdown files.

    Args:
        converted_docs: List of converted documents
        metadata: Metadata from previous steps

    Returns:
        List of Document objects
    """
    documents = []

    for doc_info in converted_docs:
        title = doc_info["title"]
        path = doc_info["path"]

        try:
            # Read the file content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create the document
            document = Document(
                title=title,
                content=content,
                metadata={
                    "source_path": path,
                    "file_type": "markdown",
                    "filename": os.path.basename(path),
                    "source": "epic_help_center",
                    "url": doc_info.get("url", ""),
                    "original_index": doc_info.get("original_index"),
                },
            )
            documents.append(document)

        except Exception as e:
            print(f"Error generating document from {path}: {str(e)}")

    print(f"Generated {len(documents)} Document objects")
    return documents


@step
def chunk_and_enrich_documents(
    documents: List[Document],
    dynamic_chunking: bool = True,
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
    apply_enrichment: bool = True,
) -> List[Document]:
    """Chunk documents and enrich with contextual information.

    Args:
        documents: List of documents to process
        dynamic_chunking: Whether to use dynamic chunking
        min_chunk_size: Minimum chunk size when using dynamic chunking
        max_chunk_size: Maximum chunk size when using dynamic chunking
        chunk_overlap: Overlap between chunks
        apply_enrichment: Whether to apply contextual enrichment

    Returns:
        Documents with chunks created and enriched
    """
    import asyncio

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

    # Apply enrichment if enabled
    if apply_enrichment:
        # Get the contextual enrichment service
        enrichment_service = container.get("contextual_enrichment_service")

        async def enrich_all():
            enriched_documents = []

            for document in processed_documents:
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
        enriched_documents = asyncio.run(enrich_all())
        print(f"Enriched chunks for {len(enriched_documents)} documents")
        return enriched_documents

    # Return documents without enrichment
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

    # Get required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    embedding_service = container.get("embedding_service")

    # Note: Enrichment should already be done in the chunk_and_enrich_documents step
    # We're just ingesting the already enriched chunks here

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
def help_center_processing_pipeline(
    json_path: str,
    output_dir: str,
    images_dir: str,
    start_index: int = 0,
    limit: Optional[int] = None,
    dynamic_chunking: bool = True,
    min_chunk_size: int = 300,
    max_chunk_size: int = 800,
    chunk_overlap: int = 50,
    apply_enrichment: bool = True,
) -> Dict[str, Any]:
    """Pipeline for processing Epic help center documentation.

    Args:
        json_path: Path to the help center JSON file
        output_dir: Directory to save markdown output
        images_dir: Directory containing images
        start_index: Index to start processing from
        limit: Optional limit on number of pages to process
        dynamic_chunking: Whether to use dynamic chunking
        min_chunk_size: Minimum chunk size when using dynamic chunking
        max_chunk_size: Maximum chunk size when using dynamic chunking
        chunk_overlap: Overlap between chunks
        apply_enrichment: Whether to apply contextual enrichment

    Returns:
        Dictionary with processing statistics
    """
    # Step 1: Load help center pages from JSON
    pages, metadata = load_help_center_pages(
        json_path=json_path, start_index=start_index, limit=limit
    )

    # Step 2: Convert pages to markdown
    converted_docs, metadata = convert_pages_to_markdown(
        pages=pages, metadata=metadata, output_dir=output_dir, images_dir=images_dir
    )

    # Step 3: Generate Document objects from markdown
    documents = generate_documents_from_markdown(
        converted_docs=converted_docs, metadata=metadata
    )

    # Step 4: Chunk and enrich documents
    processed_documents = chunk_and_enrich_documents(
        documents=documents,
        dynamic_chunking=dynamic_chunking,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        apply_enrichment=apply_enrichment,
    )

    # Step 5: Ingest documents
    ingestion_stats = ingest_documents(documents=processed_documents)

    # Since ZenML wraps our output in StepArtifact which doesn't support .get(),
    # we need to create a new dictionary with all the statistics using direct key access
    final_stats = {
        # Copy metadata stats (safely with dict access)
        "total_pages": (
            metadata["total_pages"]
            if isinstance(metadata, dict) and "total_pages" in metadata
            else 0
        ),
        "selected_pages": (
            metadata["selected_pages"]
            if isinstance(metadata, dict) and "selected_pages" in metadata
            else 0
        ),
        "start_index": (
            metadata["start_index"]
            if isinstance(metadata, dict) and "start_index" in metadata
            else 0
        ),
        "end_index": (
            metadata["end_index"]
            if isinstance(metadata, dict) and "end_index" in metadata
            else 0
        ),
        "converted_count": (
            metadata["converted_count"]
            if isinstance(metadata, dict) and "converted_count" in metadata
            else 0
        ),
        "success_rate": (
            metadata["success_rate"]
            if isinstance(metadata, dict) and "success_rate" in metadata
            else 0
        ),
        # Add ingestion stats (safely with dict access)
        "processed_document_count": (
            ingestion_stats["processed_document_count"]
            if isinstance(ingestion_stats, dict)
            and "processed_document_count" in ingestion_stats
            else 0
        ),
        "total_chunk_count": (
            ingestion_stats["total_chunk_count"]
            if isinstance(ingestion_stats, dict)
            and "total_chunk_count" in ingestion_stats
            else 0
        ),
        "avg_chunks_per_document": (
            ingestion_stats["avg_chunks_per_document"]
            if isinstance(ingestion_stats, dict)
            and "avg_chunks_per_document" in ingestion_stats
            else 0
        ),
    }

    return final_stats
