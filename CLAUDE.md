# Development Guidelines for Help Center Documentation RAG System

## Commands

### Bun/TypeScript Commands (HTML to Markdown Conversion)
- Install: `bun install`
- Run: `bun run index.ts`
- Run HTML to Markdown converter: `bun run src/scripts/json-to-markdown.ts`
- Run Markdown condenser: `bun run src/scripts/condense-markdown.ts input.md output.md`
- Run token counter: `bun run src/scripts/count-tokens.ts input.md`
- Run parallel crawler: `bun run src/scripts/parallel-json-crawler.ts`
  - With images: `bun run src/scripts/parallel-json-crawler.ts --images-dir output/images`
  - All images: `bun run src/scripts/parallel-json-crawler.ts --images-dir output/images --all-images`
  - No images: `bun run src/scripts/parallel-json-crawler.ts --no-images`
- Typecheck: `bun x tsc --noEmit`
- Format: `bun x prettier --write "**/*.{ts,js,json}"`

### Python RAG System Commands
- Python Format: `black help_rag/**/*.py`
- Python Lint: `flake8 --max-complexity 10 --max-line-length 88 help_rag/`
- Python dependencies: `uv add ...`
- Reset Database: `just reset`
- Run Evaluation: `just evaluate`
- Test Enrichment: `just enrich-simple`
- Generate Document Enrichment: `just enrich-docs`
- Process Sample Docs: `just process-samples`

### Query Testing Commands
- Basic Query: `just query "How do I reset my password?"`
- Hybrid Search: `just hybrid "How do I configure settings?"`
- BM25 Search: `just bm25 "How to export data?"`
- Full BM25 Output: `just bm25-full "How do I create a new account?"`

## Project Structure

### TypeScript Document Processing
- **src/index.ts**: Main entry point that shows available scripts
- **src/scripts/**: Contains tools for the documentation processing pipeline
  - **parallel-json-crawler.ts**: Scrapes help documentation website and outputs to JSON with images
  - **json-to-markdown.ts**: Converts scraped JSON to markdown format with local image references
  - **condense-markdown.ts**: Reduces markdown content to fit within context windows
  - **count-tokens.ts**: Estimates token counts for LLM context windows

### Python RAG System
- **help_rag/**: Main package for the RAG system
  - **application/**: Application layer with business logic
    - **pipelines/**: ZenML pipelines for document processing and evaluation
    - **use_cases/**: Core application use cases (ingest, retrieve, enrich)
  - **domain/**: Domain layer with core models and interfaces
    - **models/**: Domain models (Document, Chunk, etc.)
    - **repositories/**: Repository interfaces
    - **services/**: Service interfaces (chunking, embedding, retrieval)
  - **infrastructure/**: Infrastructure implementations
    - **config/**: System configuration
    - **container.py**: Dependency injection container
    - **document_processing/**: Document processing implementations
    - **embedding/**: Vector store implementations
    - **llm/**: LLM service implementations
  - **interfaces/**: User interfaces
    - **cli/**: Command line interface

## About Help Center Documentation

This project provides a generic system for processing help center documentation from any website. It can be configured to scrape, process, and search through:

1. User guides and tutorials
2. API documentation
3. Troubleshooting guides
4. Feature documentation
5. Configuration and setup instructions

The RAG system enhances search and retrieval by adding contextual enrichment to document chunks, improving relevance for natural language queries about any domain.

## Contextual Enrichment Approach

Our RAG system implements Anthropic's contextual retrieval methodology:

1. Documents are chunked into manageable segments
2. LLM generates a context description for each chunk
3. The context + chunk are embedded and stored in vector database
4. Retrieval quality is improved, especially for natural language queries
5. Evaluation shows improved ranking and relevance scores

## Code Style

- **TypeScript**: Use strict mode with ESNext features
- **Python**: Follow PEP 8 guidelines with a max line length of 88
- **Domain-Driven Design**: Organize code by domain concepts with clear boundaries
- **Asynchronous Programming**: Use async/await patterns in both Python and TypeScript
- **Dependency Injection**: Use container for service registration and resolution
- **Error Handling**: Proper exception handling with meaningful error messages
- **Documentation**: Docstrings for public APIs, inline comments for complex logic

