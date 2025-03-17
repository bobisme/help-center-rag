# Epic Documentation Tools

This project contains tools for working with Epic healthcare system documentation:

1. **Epic Documentation CLI**: A tool to crawl Epic docs and convert to Markdown
2. **HTML to Markdown Converter**: A Python tool for preprocessing HTML docs
3. **Epic Documentation RAG System**: A retrieval system based on Anthropic's Contextual Retrieval methodology

## Documentation CLI Features

- Crawls the Epic documentation website
- Extracts all documentation content
- Downloads screenshots and images, saving them locally
- Converts HTML to clean Markdown with local image references
- Preserves document structure and hierarchy
- Generates a comprehensive table of contents
- Outputs everything to a single searchable markdown file
- Condenses markdown to fit in LLM context windows
- Estimates token counts for different LLM models

## HTML to Markdown Converter Features

- Specialized HTML cleanup for Epic documentation
- Fixes nested list structures for proper Markdown conversion
- Converts inline styles to semantic HTML (bold/italic)
- Handles image path resolution and processing
- Well-structured Python package design
- Command-line interface with Typer and Rich

## RAG System Features

- Retrieval-Augmented Generation using Anthropic's Contextual Retrieval methodology
- Domain-driven design with clean architecture
- Dynamic document chunking optimized for context retrieval
- Two-stage retrieval process for better query accuracy
- Qdrant vector database integration
- SQLite document store with JSON support
- ZenML pipeline orchestration
- Command-line interface with Rich formatting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/epic-help.git
cd epic-help

# Install dependencies
bun install

# Install globally (optional)
bun install-global
```

## Usage

```bash
# Show all available commands
epic-help

# Get help for a specific command
epic-help help <command>
# or
epic-help <command> --help
```

### Available Commands

The CLI provides the following commands:

- **crawl**: Crawl the Epic docs website and output to JSON with images
- **convert**: Convert scraped JSON to markdown format with local image references
- **condense**: Reduce markdown content to fit within context windows
- **count**: Estimate token counts for LLM context windows

### Workflow Examples

Full workflow from crawling to tokenization:

```bash
# 1. Crawl the documentation website with images
epic-help crawl --depth 3 --concurrency 8 --images-dir output/images

# 2. Convert HTML to markdown with local image references
epic-help convert

# 3. Condense markdown content
epic-help condense output/epic-docs.md output/epic-docs-condensed.md

# 4. Count tokens for different LLM models
epic-help count output/epic-docs-condensed.md
```

For screenshots only (default):
```bash
epic-help crawl --images-dir output/images
```

For all images including icons:
```bash
epic-help crawl --all-images --images-dir output/images
```

With auto-saving for crash recovery:
```bash
epic-help crawl --images-dir output/images --autosave 300000
```

To disable image download:
```bash
epic-help crawl --no-images
```

## Command Details

### crawl

```bash
epic-help crawl [options]

Options:
  --url, -u <url>           Base URL to crawl
  --output, -o <file>       Output JSON file path
  --concurrency, -c <num>   Number of parallel workers
  --depth, -d <num>         Maximum crawl depth
  --max, -m <num>           Maximum pages to process
  --timeout, -t <ms>        Page load timeout in milliseconds
  --wait, -w <ms>           Wait time for dynamic content in milliseconds
  --interval, -i <ms>       Delay between requests in milliseconds
  --images-dir <dir>        Directory to save downloaded images
  --no-images               Disable image downloading
  --all-images              Download all images (not just screenshots)
  --autosave <ms>           Interval in milliseconds to auto-save partial results
```

### convert

```bash
epic-help convert [options]

Options:
  --input, -i <file>       Input JSON file path
  --output, -o <file>      Output markdown file path
  --metadata, -m <file>    Output metadata JSON file path
```

### condense

```bash
epic-help condense <input-file> <output-file> [options]

Options:
  --abbreviate             Enable abbreviations for greater reduction
  --summarize              Enable aggressive content summarization
```

### count

```bash
epic-help count <file-path>
```

## Requirements

- [Bun](https://bun.sh/) runtime
- Playwright browsers (installed automatically)

## Development

- Install dependencies: `bun install`
- Start the CLI tool: `bun start`
- Typecheck: `bun x tsc --noEmit`
- Format: `bun x prettier --write "**/*.{ts,js,json}"`
- Lint: `bun run lint`

## HTML to Markdown Converter Usage

The HTML to Markdown converter is optimized for Epic documentation structure.

```bash
# Basic conversion from a JSON file
python -m html2md convert --file path/to/epic/docs.json

# Convert a specific page by ID
python -m html2md convert --file path/to/epic/docs.json --page-id 12345

# List available pages in the JSON file
python -m html2md list --file path/to/epic/docs.json

# Show tool information
python -m html2md info
```

The converter provides special preprocessing for:
- Nested lists (fixing indentation issues)
- Inline styles (converting to semantic HTML)
- Image paths (resolving to local references)
- Link handling (processing local references)

## RAG System Usage

The RAG system builds on the converted documentation to provide intelligent retrieval of Epic documentation.

### Project Structure

```
epic_rag/
├── domain/                 # Core business logic and entities
│   ├── models/             # Domain entities
│   ├── repositories/       # Data access interfaces
│   └── services/           # Business logic interfaces
├── application/            # Use cases and orchestration
│   ├── pipelines/          # ZenML pipelines
│   └── use_cases/          # Application use cases
├── infrastructure/         # Technical implementations
│   ├── config/             # Configuration
│   ├── embedding/          # Vector database implementations
│   └── persistence/        # Data persistence implementations
└── interfaces/             # User interfaces
    └── cli/                # Command-line interface
```

### Contextual Retrieval Methodology

This system implements Anthropic's Contextual Retrieval approach, which improves upon standard RAG systems by:

1. **Two-Stage Retrieval**: Initial broader retrieval followed by a more focused retrieval
2. **Dynamic Chunk Sizing**: Intelligently determining chunk sizes based on content
3. **Context-Aware Merging**: Combining retrieved chunks based on semantic relatedness
4. **Relevance Filtering**: Using LLMs to filter retrieved chunks by relevance
5. **Query Transformation**: Rewriting queries to better match document corpus semantics

### Basic Usage

```bash
# Install the RAG system
pip install -e .

# Ingest documents
epic-rag ingest --source-dir data/markdown

# Query the system
epic-rag query "How do I create a new patient record?"

# Show system information
epic-rag info

# Test the embedding service
epic-rag test-embed "How do I create a new patient record?"

# Compare text similarity
epic-rag test-embed "How do I create a new patient record?" --compare "What's the process for adding a new patient?"

# Visualize document chunks
epic-rag chunks --file data/markdown/patient_registration.md --dynamic
```

### Testing Database and Embeddings

```bash
# Test the database
epic-rag test-db

# Test with the default embedding provider (HuggingFace)
epic-rag test-embed "This is a test of the embedding service"

# Test with local HuggingFace model on GPU
epic-rag test-embed "This is a test of the embedding service" --provider huggingface

# Test with a specific provider (OpenAI)
epic-rag test-embed "This is a test of the embedding service" --provider openai

# Test with Gemini embeddings
epic-rag test-embed "This is a test of the embedding service" --provider gemini

# Compare semantic similarity between texts using local model
epic-rag test-embed "Epic is a healthcare software company" --compare "Epic Systems makes EHR software for hospitals"

# Compare using Gemini
epic-rag test-embed "Epic is a healthcare software company" --compare "Epic Systems makes EHR software for hospitals" --provider gemini

# Compare using OpenAI
epic-rag test-embed "Epic is a healthcare software company" --compare "Epic Systems makes EHR software for hospitals" --provider openai
```

### Environment Variables

The system uses the following environment variables:

```
# API Keys for embeddings (choose one or both)
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key

# Embedding configuration
EPIC_RAG_EMBEDDING_PROVIDER=huggingface  # or openai, gemini
EPIC_RAG_EMBEDDING_API_KEY=your_embedding_api_key  # for cloud providers

# Provider-specific model configuration
EPIC_RAG_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EPIC_RAG_GEMINI_EMBEDDING_MODEL=gemini-embedding-exp-03-07
EPIC_RAG_HUGGINGFACE_MODEL=intfloat/e5-large-v2

# HuggingFace specific settings (for local embeddings)
EPIC_RAG_EMBEDDING_DEVICE=cuda  # cuda, cpu, mps

# Qdrant configuration (optional for remote Qdrant)
EPIC_RAG_QDRANT_URL=https://your-qdrant-instance.com
EPIC_RAG_QDRANT_API_KEY=your_qdrant_api_key
EPIC_RAG_QDRANT_COLLECTION=epic_docs
```

### ZenML Pipelines

```bash
# Run document processing pipeline
python -m epic_rag.application.pipelines.document_processing_pipeline \
  --source-dir data/markdown \
  --dynamic-chunking

# Run query evaluation pipeline
python -m epic_rag.application.pipelines.query_evaluation_pipeline \
  --query-file data/test_queries.txt
```

## License

MIT