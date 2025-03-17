# Epic Documentation RAG System Plan

This document outlines the plan for building a Retrieval-Augmented Generation (RAG) system based on Epic documentation using Anthropic's Contextual Retrieval methodology.

## System Overview

The system will process the converted Epic documentation (from HTML to Markdown) and create a retrieval system that can provide accurate, contextual responses to queries about Epic systems. The architecture follows domain-driven design principles to ensure maintainability and separation of concerns.

### Key Components

1. **Data Processing Layer**: Transforms the Markdown documents into suitable formats for embedding and retrieval
2. **Vector Store**: Qdrant for storing and retrieving document embeddings
3. **Document Store**: SQLite with JSON extension for storing document metadata and content
4. **Retrieval Engine**: Implementation of Contextual Retrieval methodology for fetching relevant document chunks
5. **MLOps Pipeline**: ZenML for orchestrating the end-to-end workflow
6. **Evaluation Framework**: Metrics and tools to assess system performance

## Anthropic's Contextual Retrieval Methodology

The system implements Anthropic's Contextual Retrieval approach, which improves upon standard RAG systems by:

1. **Two-Stage Retrieval**: Initial broader retrieval followed by a more focused retrieval based on the query context
2. **Dynamic Chunk Sizing**: Intelligently determining the appropriate chunk size during document processing
3. **Context-Aware Merging**: Combining retrieved chunks based on their semantic relatedness
4. **Relevance Filtering**: Using an LLM to filter and rank retrieved chunks for relevance
5. **Query Transformation**: Rewriting queries to better match the document corpus semantics

## Technology Stack

- **Python**: Core implementation language
- **ZenML**: MLOps and pipeline orchestration
- **Qdrant**: Vector database for embeddings
- **SQLite + JSON Extension**: Document metadata and storage
- **Embedding Models**: To be determined (potentially OpenAI, Cohere, or open-source alternatives)
- **LLM Integration**: Abstract interface supporting multiple providers

## TODO Checklist

### Phase 1: Foundation and Data Processing

- [x] Set up project structure following domain-driven design
- [x] Create core domain models and interfaces
- [x] Implement document processing for dynamic chunking
- [x] Set up SQLite database with JSON extension
- [x] Implement document storage and retrieval layer

### Phase 2: Vector Database and Embeddings

- [x] Research and select appropriate embedding model
- [x] Set up Qdrant instance (local/cloud)
- [x] Implement embedding generation pipeline
- [x] Create vector indexing and retrieval functionality
- [x] Add support for multiple embedding providers (HuggingFace, OpenAI, Gemini)
- [x] Implement local GPU-accelerated embedding with E5-large-v2
- [x] Design and implement caching strategy for embeddings

### Phase 3: Retrieval Engine

- [x] Implement first-stage broad retrieval
- [x] Develop query transformation functionality using local LLM (Ollama)
- [x] Build context-aware chunk merging logic
- [x] Create relevance filtering using similarity scores
- [x] Implement BM25 lexical search for exact keyword matching
  - [x] Standard implementation using rank-bm25
  - [x] High-performance implementation using Huggingface's bm25s
- [x] Create rank fusion to combine vector and BM25 search results
- [x] Implement reranking with cross-encoder models for improved result relevance
- [x] Develop retrieval pipeline with Anthropic's methodology

### Phase 4: ZenML Pipeline Integration

- [x] Define ZenML pipeline components
- [x] Create data ingestion step
- [x] Implement document processing steps
- [x] Build embedding generation step
- [x] Set up retrieval evaluation step
- [x] Create end-to-end orchestration pipeline
- [x] Implement contextual enrichment for chunks using LLM
  - [x] Create prompt template for generating chunk context
  - [x] Integrate with local LLM (Ollama/Gemma) for context generation
  - [x] Add context to chunks before embedding

### Phase 5: Evaluation and Optimization

- [ ] Design evaluation metrics (e.g., retrieval precision, relevance scoring)
- [ ] Create test query dataset from Epic documentation
- [ ] Implement evaluation pipeline
- [ ] Compare against baseline approaches
- [ ] Optimize retrieval parameters based on evaluation results

### Phase 6: Deployment and Monitoring

- [ ] Containerize the application
- [ ] Set up model and data versioning
- [ ] Implement monitoring for system performance
- [ ] Create deployment pipeline for continuous updates
- [ ] Develop user interface for querying the system

## Initial Implementation Plan

1. Start with domain model definition
2. Implement the data processing layer
3. Set up the document store
4. Integrate with Qdrant
5. Implement the core retrieval logic
6. Create ZenML pipelines
7. Develop evaluation framework
8. Build deployment mechanisms

## Design Principles

1. **Modularity**: Components should be loosely coupled
2. **Testability**: All core logic should be easily testable
3. **Extensibility**: System should support different embedding models and LLMs
4. **Observability**: Performance metrics should be trackable
5. **Maintainability**: Code should follow clean code practices and include documentation