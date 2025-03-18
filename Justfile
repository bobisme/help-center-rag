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

# Test image description generation
image-describe:
    python image_description_demo.py --sample-limit 5

# Test image description with a specific document
image-describe-doc doc="test/samples/renew-a-certificate.md":
    python image_description_demo.py --doc-path "{{doc}}"

# Test image-enhanced contextual enrichment
image-enrich doc="test/samples/renew-a-certificate.md":
    python image_enrichment_demo.py "{{doc}}"

# Test image enrichment with custom image size
image-enrich-custom doc="test/samples/renew-a-certificate.md" size="128":
    python image_enrichment_demo.py "{{doc}}" --min-image-size "{{size}}"

# Show full enriched chunks for a document
show-chunks doc="test/samples/renew-a-certificate.md":
    python show_enriched_chunks.py "{{doc}}"

# Show full enriched chunks with custom parameters
show-chunks-custom doc="test/samples/renew-a-certificate.md" chunk_size="600" overlap="50":
    python show_enriched_chunks.py "{{doc}}" --chunk-size "{{chunk_size}}" --chunk-overlap "{{overlap}}"

# Run standalone image description demo
image-standalone doc="test/samples/renew-a-certificate.md":
    python image_description_standalone.py "{{doc}}"

# Run SmolVLM image description demo
smolvlm-describe:
    python smolvlm_image_description_demo.py --sample-limit 5

# Run SmolVLM image description with a specific document
smolvlm-describe-doc doc="test/samples/renew-a-certificate.md":
    python smolvlm_image_description_demo.py --doc-path "{{doc}}"

# Compare image descriptions from Gemma and SmolVLM
compare-descriptions doc="test/samples/renew-a-certificate.md" limit="3":
    python compare_image_descriptions.py --doc-path "{{doc}}" --sample-limit {{limit}}

# Process help center documents using the custom script
process-help-center count="10":
    python process_help_center.py pipeline --output-dir data/help_center --limit {{count}}

# List help center documents
list-help-center limit="20":
    python process_help_center.py list --limit {{limit}}

# Process all help center documents (this will take a while)
process-all-help-center:
    python process_help_center.py pipeline --output-dir data/help_center

# Process help center documents using pipeline components
run-help-center count="10":
    python run_help_center_pipeline.py --output-dir data/help_center --limit {{count}}

# Process all help center documents using pipeline components
run-help-center-all:
    python run_help_center_pipeline.py --output-dir data/help_center

# Process help center documents without enrichment
run-help-center-no-enrichment count="10":
    python run_help_center_pipeline.py --output-dir data/help_center --limit {{count}} --no-enrichment

# Test query against help center documents
query-help-center q:
    python -m epic_rag.interfaces.cli.main query "{{q}}" --show-details

# BM25 search help center documents
bm25-help-center q:
    python -m epic_rag.interfaces.cli.main bm25 "{{q}}" --full-content

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
