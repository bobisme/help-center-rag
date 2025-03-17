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
    python -m epic_rag.interfaces.cli.main transform-query "How do I schedule a patient visit?"

# Test various query transformations with different models
transform-scheduling:
    python -m epic_rag.interfaces.cli.main transform-query "How do I schedule a patient visit?" --model gemma3:27b

transform-allergies:
    python -m epic_rag.interfaces.cli.main transform-query "How do I modify patient allergies?" --model gemma3:27b

transform-labs:
    python -m epic_rag.interfaces.cli.main transform-query "What are the steps to order a lab test?" --model gemma3:27b

transform-reactions:
    python -m epic_rag.interfaces.cli.main transform-query "How to document adverse reactions to medications?" --model gemma3:27b

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
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir data/markdown --pipeline document_processing --limit 5 --apply-enrichment

# Test document processing pipeline without contextual enrichment
zenml-docs-plain:
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir data/markdown --pipeline document_processing --limit 5 --skip-enrichment

# Test orchestration pipeline with contextual enrichment
zenml-orchestration-enriched:
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir data/markdown --pipeline orchestration --limit 3 --apply-enrichment

# Test contextual enrichment with sample document
enrich-test:
    #!/usr/bin/env bash
    echo "Testing contextual enrichment on sample document..."
    # Create test directory if it doesn't exist
    mkdir -p test/samples
    
    # Create a sample markdown file if it doesn't exist
    if [ ! -f test/samples/test_doc.md ]; then
      echo "# Epic Documentation Sample
    
    ## Introduction
    
    This is a sample document about Epic healthcare software documentation.
    
    ## Patient Registration
    
    The patient registration module allows healthcare providers to register new patients,
    update demographic information, and manage patient records efficiently.
    
    ## Medication Management
    
    Epic's medication management features help providers:
    
    1. Prescribe medications safely
    2. Check for drug interactions
    3. Monitor medication adherence
    4. Document adverse reactions
    " > test/samples/test_doc.md
    fi
    
    # Run the document processing pipeline with enrichment enabled
    python -m epic_rag.interfaces.cli.main zenml-run --source-dir test/samples --pipeline document_processing --apply-enrichment

# Simple test that directly shows enrichment results
enrich-simple:
    python -m epic_rag.interfaces.cli.main test-enrichment test_sample.md

# Test enrichment with custom file
enrich:
    python -m epic_rag.interfaces.cli.main test-enrichment test_sample.md --max-chunks 3

# Evaluate the impact of contextual enrichment on retrieval quality
evaluate-enrichment:
    python -m epic_rag.interfaces.cli.main evaluate-enrichment test_sample.md

# Run evaluation of contextual enrichment impact
evaluate:
    python manual_evaluation.py

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
