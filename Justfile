set shell := ["bash", "-c"]

fmt:
  black "epic_rag/**/*.py"

lint:
  flake8 --max-complexity 10 --max-line-length 88 epic_rag/

# Reset the databases and ingest markdown documents
reset:
    #!/usr/bin/env bash
    echo "Removing existing databases..."
    rm -rf data/epic_rag.db qdrant_data
    
    echo "Creating data directories if they don't exist..."
    mkdir -p data/markdown data/output
    
    echo "Ingesting markdown documents..."
    python -m epic_rag.interfaces.cli.main ingest --source-dir data/markdown --pattern "*.md"
    
    echo "Database reset complete."

# Test query transformation with default model
transform-test:
    python -m epic_rag.interfaces.cli.main transform-query "How do I set up faxing for my agency?"

# Test various query transformations with different models
transform-email:
    python -m epic_rag.interfaces.cli.main transform-query "How do I access my email in Epic?" --model gemma3:27b

transform-quotes:
    python -m epic_rag.interfaces.cli.main transform-query "How do I compare insurance quotes for a client?" --model gemma3:27b

transform-certificate:
    python -m epic_rag.interfaces.cli.main transform-query "What are the steps to renew a certificate?" --model gemma3:27b

transform-vinlink:
    python -m epic_rag.interfaces.cli.main transform-query "How do I set up VINlink Decoder for my account?" --model gemma3:27b

# Test full query with transformed queries
query q:
    python -m epic_rag.interfaces.cli.main query "{{q}}" --show-details

# Test BM25 search
bm25 q:
    python -m epic_rag.interfaces.cli.main bm25 "{{q}}"

# Test BM25 search with full content display
bm25-full q:
    python -m epic_rag.interfaces.cli.main bm25 "{{q}}" --full-content

# Test hybrid search with both BM25 and vector search
hybrid q:
    python -m epic_rag.interfaces.cli.main hybrid-search "{{q}}" --rerank

# Test hybrid search with detailed output
hybrid-full q:
    python -m epic_rag.interfaces.cli.main hybrid-search "{{q}}" --show-separate --full-content

# Test hybrid search with custom weights
hybrid-weights q b v:
    python -m epic_rag.interfaces.cli.main hybrid-search "{{q}}" --bm25-weight {{b}} --vector-weight {{v}}

# Benchmark BM25 implementations
benchmark-bm25 q:
    python -m epic_rag.interfaces.cli.main benchmark-bm25 "{{q}}" --iterations 20 --documents 100

# Benchmark BM25 implementations with custom parameters
benchmark-bm25-custom q iters docs:
    python -m epic_rag.interfaces.cli.main benchmark-bm25 "{{q}}" --iterations {{iters}} --documents {{docs}}

# Show system information
info:
    python -m epic_rag.interfaces.cli.main info

# Test document processing pipeline with contextual enrichment
zenml-docs-enriched:
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir test/samples --pipeline document_processing --limit 5 --apply-enrichment

# Test document processing pipeline without contextual enrichment
zenml-docs-plain:
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir test/samples --pipeline document_processing --limit 5 --skip-enrichment

# Test orchestration pipeline with contextual enrichment
zenml-orchestration-enriched:
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir test/samples --pipeline orchestration --limit 3 --apply-enrichment

# Process all sample documents with contextual enrichment
process-samples:
    #!/usr/bin/env bash
    echo "Processing sample documents with contextual enrichment..."
    
    # Run the document processing pipeline with enrichment enabled on all samples
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir test/samples --pipeline document_processing --apply-enrichment

# Simple test that directly shows enrichment results
enrich-simple:
    python -m epic_rag.interfaces.cli.main test-enrichment test/samples/email.md

# Test enrichment with custom file
enrich:
    python -m epic_rag.interfaces.cli.main test-enrichment test/samples/email.md --max-chunks 3

# Evaluate the impact of contextual enrichment on retrieval quality
evaluate-enrichment:
    python -m epic_rag.interfaces.cli.main evaluate-enrichment test/samples/quote-results.md

# Run evaluation of contextual enrichment impact
evaluate:
    python manual_evaluation.py

# Generate enriched contexts for sample insurance docs
enrich-insurance:
    python insurance_enrichment.py

# Run interactive demo of contextual enrichment
demo:
    python demo_enrichment.py

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
