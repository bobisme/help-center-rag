set shell := ["bash", "-c"]

fmt:
  black "help_rag/**/*.py"

lint:
  flake8 --max-complexity 10 --max-line-length 88 help_rag/

# Run all tests
test:
  python -m pytest

# Run tests with verbose output
test-v:
  python -m pytest -v

# Run html2md tests only
test-html2md:
  python -m pytest html2md/test/ -v

# Reset the databases and ingest markdown documents
reset:
    #!/usr/bin/env bash
    echo "Removing existing databases..."
    rm -rf data/help_rag.db qdrant_data
    echo "Creating data directories if they don't exist..."
    mkdir -p data/markdown data/output
    echo "Database reset complete."

# Test query transformation with default model
transform-test:
    python -m help_rag.interfaces.cli.main transform-query "How do I configure notifications?"

# Test various query transformations with different models
transform-email:
    python -m help_rag.interfaces.cli.main transform-query "How do I access my email?" --model gemma3:27b

transform-quotes:
    python -m help_rag.interfaces.cli.main transform-query "How do I compare different options?" --model gemma3:27b

transform-certificate:
    python -m help_rag.interfaces.cli.main transform-query "What are the steps to renew my subscription?" --model gemma3:27b

transform-vinlink:
    python -m help_rag.interfaces.cli.main transform-query "How do I set up integrations for my account?" --model gemma3:27b

# Test full query with transformed queries
query q:
    ./rag query "{{q}}" --show-details

# Test BM25 search
bm25 q:
    ./rag query "{{q}}" --bm25-only

# Test BM25 search with full content display
bm25-full q:
    ./rag query "{{q}}" --bm25-only --full-content

# Test hybrid search with both BM25 and vector search
hybrid q:
    ./rag query "{{q}}" --rerank

# Test hybrid search with detailed output
hybrid-full q:
    ./rag query "{{q}}" --full-content --show-details

# Test hybrid search with custom weights
hybrid-weights q b v:
    ./rag query "{{q}}" --bm25-weight {{b}} --vector-weight {{v}}

# Benchmark BM25 implementations (using query with BM25 option)
benchmark-bm25 q:
    ./rag query "{{q}}" --bm25-only

# Benchmark BM25 implementations with custom parameters
benchmark-bm25-custom q iters docs:
    ./rag query "{{q}}" --bm25-only --top-k {{docs}}

# Show system information
info:
    ./rag info
    
# Show database statistics
db-info:
    ./rag db info
    
# Clean up orphaned chunks
db-cleanup:
    ./rag db cleanup-orphans
    
# Backup database
db-backup:
    ./rag db backup
    
# Vacuum database
db-vacuum:
    ./rag db vacuum

# List all documents
db-list:
    ./rag db list-documents

# Inspect a document by title
db-inspect title:
    ./rag db inspect-document --title "{{title}}" --chunks --metadata

# Process test samples with documents ingest command
process-samples-enriched:
    ./rag documents ingest --source-dir test/samples --limit 5 --dynamic-chunking

# Process test samples with fixed chunking
process-samples-fixed:
    ./rag documents ingest --source-dir test/samples --limit 5 --fixed-chunking

# Process test samples with different chunk sizes
process-samples-custom:
    ./rag documents ingest --source-dir test/samples --limit 3 --min-chunk-size 400 --max-chunk-size 1000

# Process all sample documents
process-samples:
    #!/usr/bin/env bash
    echo "Processing all sample documents..."
    
    # Run the document ingest command on all samples
    ./rag documents ingest --source-dir test/samples

# Simple test that directly shows enrichment results
enrich-simple:
    python -m help_rag.interfaces.cli.main test-enrichment test/samples/email.md
    
# Test enrichment on a document by ID (use ID from previous output)
test-enrichment id:
    python fix_by_sql.py enrich --id "{{id}}"

# Test enrichment with custom file
enrich:
    python -m help_rag.interfaces.cli.main test-enrichment test/samples/email.md --max-chunks 3

# Evaluate the impact of contextual enrichment on retrieval quality
evaluate-enrichment:
    python -m help_rag.interfaces.cli.main evaluate-enrichment test/samples/quote-results.md

# Run evaluation of contextual enrichment impact
evaluate:
    python -m help_rag.interfaces.cli.main manual-evaluation

# Generate enriched contexts for sample docs
enrich-docs:
    python -m help_rag.interfaces.cli.main testing enrichment-demo

# Run interactive demo of contextual enrichment
demo:
    python -m help_rag.interfaces.cli.main enrichment-demo

# Test image description generation
image-describe:
    python -m help_rag.interfaces.cli.main image-describe --sample-limit 5

# Test image description with a specific document
image-describe-doc doc="test/samples/renew-a-certificate.md":
    python -m help_rag.interfaces.cli.main image-describe --doc-path "{{doc}}"

# Test image-enhanced contextual enrichment
image-enrich doc="test/samples/renew-a-certificate.md":
    python -m help_rag.interfaces.cli.main images describe --doc-path "{{doc}}"

# Test image enrichment with custom image size
image-enrich-custom doc="test/samples/renew-a-certificate.md" size="128":
    python -m help_rag.interfaces.cli.main images describe --doc-path "{{doc}}" --min-size "{{size}}"

# Show full enriched chunks for a document
show-chunks doc="test/samples/renew-a-certificate.md":
    python -m help_rag.interfaces.cli.main vis chunks --title "Certificate" --context-only

# Show full enriched chunks with custom parameters
show-chunks-custom doc="test/samples/renew-a-certificate.md" chunk_size="600" overlap="50":
    python -m help_rag.interfaces.cli.main vis chunks --title "Certificate" --limit 10 --metadata

# Run standalone image description demo
image-standalone doc="test/samples/renew-a-certificate.md":
    python -m help_rag.interfaces.cli.main images describe --doc-path "{{doc}}"

# Run SmolVLM image description demo
smolvlm-describe:
    python -m help_rag.interfaces.cli.main smolvlm-describe --sample-limit 5

# Run SmolVLM image description with a specific document
smolvlm-describe-doc doc="test/samples/renew-a-certificate.md":
    python -m help_rag.interfaces.cli.main smolvlm-describe --doc-path "{{doc}}"

# Compare image descriptions from Gemma and SmolVLM
compare-descriptions doc="test/samples/renew-a-certificate.md" limit="3":
    python -m help_rag.interfaces.cli.main compare-descriptions --doc-path "{{doc}}" --sample-limit {{limit}}
    
# Show chunks for a specific document
show-doc-chunks title:
    python -m help_rag.interfaces.cli.main show-doc-chunks "{{title}}"
    
# Show chunks for a document with metadata
show-doc-chunks-meta title:
    python -m help_rag.interfaces.cli.main show-doc-chunks "{{title}}" --metadata
    
# Show only the context added to chunks for a document
show-doc-context title:
    python -m help_rag.interfaces.cli.main show-doc-chunks "{{title}}" --context-only
    
# Show specific document by ID
show-doc-by-id id:
    python -m help_rag.interfaces.cli.main show-doc-by-id "{{id}}" --context-only

# Process help center documents using the CLI
process-help-center count="10":
    python -m help_rag.interfaces.cli.main process-help-center --output-dir data/help_center --limit {{count}}

# List help center documents
list-help-center limit="20":
    python -m help_rag.interfaces.cli.main list-help-center --limit {{limit}}

# Process all help center documents (this will take a while)
process-all-help-center:
    python -m help_rag.interfaces.cli.main process-help-center --output-dir data/help_center

# Process help center documents using pipeline components
run-help-center count="10":
    python -m help_rag.interfaces.cli.main run-help-center --output-dir data/help_center --limit {{count}}

# Process all help center documents using pipeline components
run-help-center-all:
    python -m help_rag.interfaces.cli.main run-help-center --output-dir data/help_center

# Process help center documents without enrichment
run-help-center-no-enrichment count="10":
    python -m help_rag.interfaces.cli.main run-help-center --output-dir data/help_center --limit {{count}} --no-enrichment

# Process help center documents using ZenML pipeline
zenml-help-center count="10":
    python -m help_rag.interfaces.cli.main zenml-run --pipeline help_center --limit {{count}} --apply-enrichment

# Process help center documents using ZenML pipeline without enrichment
zenml-help-center-no-enrichment count="10":
    python -m help_rag.interfaces.cli.main zenml-run --pipeline help_center --limit {{count}} --skip-enrichment

# Test query against help center documents
query-help-center q:
    ./rag query "{{q}}" --show-details

# BM25 search help center documents
bm25-help-center q:
    ./rag query "{{q}}" --bm25-only --full-content

# Ask a question and get an answer using RAG
ask q:
    ./rag ask "{{q}}"

# Ask a question and see the context used
ask-with-context q:
    ./rag ask "{{q}}" --show-context

# Ask a question and see detailed metrics
ask-with-metrics q:
    ./rag ask "{{q}}" --show-metrics

# Ask a question with custom temperature
ask-temp q temp="0.5":
    ./rag ask "{{q}}" --temperature {{temp}}
    
# Ask with lower relevance threshold to get more context
ask-broad q:
    ./rag ask "{{q}}" --min-relevance 0.3 --top-k 8

# Ask with verbose output to see what's happening
ask-debug q:
    ./rag ask "{{q}}" --verbose --show-metrics --min-relevance 0.3

# Test help center pipeline with a small sample using non-ZenML implementation
test-help-center:
    python test_help_center_pipeline.py --limit 5

# Test help center pipeline without enrichment
test-help-center-no-enrichment:
    python test_help_center_pipeline.py --limit 5 --no-enrichment

# Stop ZenML server
zenml-stop:
    zenml logout --local

# Start ZenML server
zenml-start:
    #!/usr/bin/env bash
    # Conditional command for macOS vs other platforms
    if [ "$(uname)" = "Darwin" ]; then
        export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
        zenml login --local
    else
        zenml login --local
    fi
