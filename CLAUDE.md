# Development Guidelines for Epic Documentation RAG System

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
- Python Format: `black epic_rag/**/*.py`
- Python Lint: `flake8 --max-complexity 10 --max-line-length 88 epic_rag/`
- Python dependencies: `uv add ...`
- Reset Database: `just reset`
- Run Evaluation: `just evaluate`
- Test Enrichment: `just enrich-simple`
- Generate Insurance Enrichment: `just enrich-insurance`
- Process Sample Docs: `just process-samples`

### Query Testing Commands
- Basic Query: `just query "How do I access my email in Epic?"`
- Hybrid Search: `just hybrid "How do I compare insurance quotes?"`
- BM25 Search: `just bm25 "How to renew a certificate?"`
- Full BM25 Output: `just bm25-full "How do I set up faxing?"`

## Project Structure

### TypeScript Document Processing
- **src/index.ts**: Main entry point that shows available scripts
- **src/scripts/**: Contains tools for the documentation processing pipeline
  - **parallel-json-crawler.ts**: Scrapes Epic docs website and outputs to JSON with images
  - **json-to-markdown.ts**: Converts scraped JSON to markdown format with local image references
  - **condense-markdown.ts**: Reduces markdown content to fit within context windows
  - **count-tokens.ts**: Estimates token counts for LLM context windows

### Python RAG System
- **epic_rag/**: Main package for the RAG system
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

## About Epic Help Documentation

This project is focused on the Applied Epic insurance agency management system documentation. Applied Epic is a comprehensive system used by insurance agencies to manage:

1. Client and policy information
2. Quotes and proposals
3. Certificates and proofs of insurance
4. Agency communications (email, fax)
5. Accounting and billing operations

The RAG system enhances search and retrieval by adding contextual enrichment to document chunks, improving relevance for natural language queries about insurance operations.

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

