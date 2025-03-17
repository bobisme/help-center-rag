set shell := ["bash", "-c"]

fmt:
  black epic_rag/**/*.py

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


# Show system information
info:
    python -m epic_rag.interfaces.cli.main info

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
