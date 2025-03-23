# ZenML Feature Engineering Pipeline for Epic Documentation

This document describes the ZenML pipeline implementation for the Epic Documentation RAG system.

## Pipeline Overview

The Feature Engineering pipeline is a comprehensive workflow that processes documentation from HTML to vector database storage in the following steps:

1. `convert_to_markdown`: Converts HTML content to Markdown format
2. `chunk_document`: Creates optimally sized chunks from the Markdown documents
3. `add_document_context`: Adds context to each chunk to enhance retrieval
4. `add_image_descriptions`: Generates descriptions for images in the documents
5. `load_to_document_store`: Saves documents and chunks to the SQLite document store
6. `load_to_vector_db`: Embeds chunks and stores them in the Qdrant vector database

## Implementation Details

### Directory Structure

```
epic_rag/
├── pipelines/
│   ├── __init__.py
│   └── feature_engineering.py
└── steps/
    ├── __init__.py
    ├── convert_to_markdown.py
    ├── chunk_document.py
    ├── add_document_context.py
    ├── add_image_descriptions.py
    ├── load_to_document_store.py
    └── load_to_vector_db.py
```

### Step Implementation Patterns

Each step follows a consistent pattern:

1. **Decorated Function**: Each step is a Python function decorated with `@step`
2. **Type Annotations**: Clear input/output typing for ZenML to track
3. **Dependency Injection**: Uses the container to access required services
4. **Async Processing**: Utilizes asyncio for efficient document processing
5. **Batch Support**: All steps support processing multiple documents
6. **Error Handling**: Robust error handling to continue processing despite failures
7. **Logging**: Detailed logging using ZenML's logger

### Pipeline Parameters

The feature engineering pipeline supports the following parameters:

- **Document Selection**:
  - `index`: Process a specific document by index
  - `offset`: Starting index for batch processing
  - `limit`: Maximum number of documents to process
  - `all_docs`: Whether to process all documents

- **Source Configuration**:
  - `source_path`: Path to the source JSON file
  - `images_dir`: Directory where images are stored

- **Chunking Configuration**:
  - `min_chunk_size`: Minimum chunk size in characters
  - `max_chunk_size`: Maximum chunk size in characters
  - `chunk_overlap`: Overlap between chunks in characters
  - `dynamic_chunking`: Whether to use dynamic chunking

- **Processing Options**:
  - `skip_enrichment`: Whether to skip contextual enrichment
  - `skip_image_descriptions`: Whether to skip image description generation
  - `dry_run`: Whether to skip saving to the database

### CLI Integration

The pipeline is accessible through the CLI:

```bash
# Process all documents
epic-rag pipeline feature-engineering

# Process a single document
epic-rag pipeline feature-engineering --index 0

# Process a batch of documents
epic-rag pipeline feature-engineering --offset 0 --limit 10

# Test without saving to database
epic-rag pipeline feature-engineering --dry-run

# Customize processing
epic-rag pipeline feature-engineering --no-enrich --no-images
```

## Testing

A test script (`test_zenml_pipeline.py`) is provided to verify the pipeline's functionality:

```bash
# Run the test
python test_zenml_pipeline.py
```

The test runs the pipeline on a single document with a dry run and skips image descriptions to provide a quick verification.

## ZenML Benefits

Using ZenML for pipeline orchestration provides several advantages:

1. **Dependency Management**: Clearly defined inputs and outputs between steps
2. **Caching**: Avoid re-running expensive steps when inputs haven't changed
3. **Versioning**: Track changes to pipelines and their components
4. **Monitoring**: Observe pipeline execution and performance
5. **Scalability**: Run pipelines on different compute environments

## Future Enhancements

Potential improvements for the pipeline include:

1. **Parallel Processing**: Implement true parallel processing for document batches
2. **Step Materialization**: Store intermediate outputs in ZenML artifact store 
3. **Pipeline Visualization**: Use ZenML dashboard for pipeline visualization
4. **Custom Artifacts**: Define custom ZenML artifact types
5. **Pipeline Metadata**: Track additional metadata about pipeline runs