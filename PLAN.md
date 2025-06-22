# Applied Epic Documentation RAG System Plan

This document outlines the plan for building a Retrieval-Augmented Generation (RAG) system based on Applied Epic insurance agency management system documentation using Anthropic's Contextual Retrieval methodology.

## System Overview

The system will process the converted Applied Epic documentation (from HTML to Markdown) and create a retrieval system that can provide accurate, contextual responses to queries about insurance agency operations, policy management, quoting, certificates, and other insurance workflows. The architecture follows domain-driven design principles to ensure maintainability and separation of concerns.

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

- [x] Design evaluation metrics (e.g., retrieval precision, relevance scoring)
  - [x] Implement Recall@k, Precision@k, MRR, and NDCG metrics
  - [x] Add Anthropic's failure rate reduction metric (1 - recall@20)
- [x] Create test query dataset from Applied Epic insurance documentation
  - [x] Develop LLM-based dataset generator for insurance-related query-document pairs
  - [x] Implement ground truth tracking for evaluation
  - [x] Update evaluation to use real insurance agency system documents
- [x] Implement evaluation pipeline
  - [x] Create comparative evaluation of standard vs. enriched retrieval
  - [x] Build detailed metrics reporting and visualization
  - [x] Measure performance impact of contextual enrichment
- [x] Compare against baseline approaches
  - [x] Side-by-side comparison of retrieval with/without enrichment
  - [x] Measure latency and quality trade-offs
  - [x] Implement manual evaluation script for direct performance comparison
- [x] Optimize retrieval parameters based on evaluation results
  - [x] Confirmed value of contextual enrichment with 0.20 position improvement in relevant result ranking
  - [x] Measured impact of enrichment on search relevance scoring
  - [x] Documented performance improvements using Anthropic's recommended metrics

### Phase 6: Image Enrichment for Enhanced Context

- [x] Implement image description generation using Gemma 27B model
  - [x] Create image processing pipeline to extract image context from markdown
  - [x] Develop prompt template for Gemma 27B to generate image descriptions
  - [x] Build async processing mechanism for batch image description generation
  - [x] Create storage mechanism for image descriptions
  - [x] Integrate image descriptions into contextual enrichment process
- [x] Add support for alternative image description models
  - [x] Implement SmolVLM-Synthetic as a lightweight vision-language model
  - [x] Create swappable image description service interface
  - [x] Build configuration options to select the active model
  - [x] Create comparison scripts to evaluate model outputs
- [x] Enhance chunking to preserve image context
  - [x] Modify chunking service to keep images with surrounding text
  - [x] Implement special handling for image-rich sections
  - [x] Create image reference tracking during chunking
- [x] Extend contextual enrichment with image descriptions
  - [x] Update enrichment prompts to incorporate image descriptions
  - [x] Design method to weigh text vs. image context based on content type
  - [x] Implement evaluation framework to measure impact of image descriptions
  - [x] Add metrics specific to image-enhanced retrieval

### Phase 7: Help Center Document Processing Pipeline

- [x] Develop batch processing pipeline for Help Center JSON data
  - [x] Create script to extract and list all pages from epic-docs.json
  - [x] Build a batch converter to process all pages to markdown
  - [x] Set up automatic image path resolution for converted pages
  - [x] Create output directory structure for processed documents
  - [x] Generate metadata for processed documents (source page, category, etc.)
- [x] Implement document ingestion pipeline
  - [x] Create batch ingestion script to process all markdown documents
  - [x] Set up chunking with image context preservation
  - [x] Implement contextual enrichment with image descriptions
  - [x] Configure vector embedding for all processed chunks
  - [x] Store processed chunks in document database and vector store
- [x] Implement ZenML pipeline for Help Center processing
  - [x] Create step to load and parse JSON help center documents
  - [x] Implement HTML to markdown conversion step
  - [x] Add document generation and processing steps
  - [x] Implement chunking and enrichment in the pipeline
  - [x] Add embedding and storage steps to complete the pipeline
  - [x] Add caching and rich progress reporting
- [ ] Optimize retrieval for Help Center content
  - [ ] Add domain-specific query transformation prompts
  - [ ] Calibrate BM25 and vector weights for insurance documentation
  - [ ] Implement result filtering based on document categories
  - [ ] Add metadata boosts for query-relevant document types
- [ ] Add evaluation metrics for Help Center content
  - [ ] Create test query set based on common insurance agency questions
  - [ ] Implement relevance scoring specific to insurance documentation
  - [ ] Compare retrieval with and without image enrichment
  - [ ] Generate benchmarks for retrieval performance

### Phase 8: Data Management and Versioning

- [x] Implement document overwrite functionality in pipeline
  - [x] Add logic to detect existing documents by epic_page_id
  - [x] Create functionality to delete old document versions before saving new ones
  - [x] Update the ingest step to handle document replacement
  - [x] Ensure all associated chunks are properly cleaned up during replacement
  - [x] Add performance optimization to batch delete operations
- [x] Add database maintenance utilities
  - [x] Create CLI command to clean up orphaned chunks
  - [x] Implement database backup functionality
  - [x] Add database compression/vacuum operations
  - [x] Create document metadata inspection tools
- [x] Improve pipeline statistics tracking
  - [x] Add metrics for document replacement vs. new documents
  - [x] Track version history counts
  - [x] Implement detailed logging for data operations

### Phase 9: Question Answering with LLM Integration

- [x] Implement LLM service for answering questions based on retrieved chunks
  - [x] Design abstract LLM service interface for answer generation
  - [x] Implement integration with Ollama LLM provider
  - [x] Add configuration options for model selection and parameters
  - [x] Implement prompt template system for answer generation
- [x] Create prompt engineering framework for effective RAG answers
  - [x] Design base prompt template with context insertion points
  - [x] Implement chunk formatting for optimal context utilization
  - [x] Create answer constraints to ensure response quality
  - [x] Add system for handling citations and references to source material
- [x] Build end-to-end query-to-answer pipeline
  - [x] Create pipeline that connects retrieval to answer generation
  - [x] Add fallback mechanisms for handling retrieval failures
  - [x] Build in safeguards against hallucination and incorrect information
  - [ ] Implement query classification to determine appropriate response strategies
- [ ] Add evaluation metrics for answer quality
  - [ ] Implement accuracy metrics comparing answers to ground truth
  - [ ] Create system for detecting hallucinations in generated answers
  - [ ] Add human evaluation interface for subjective quality assessment
  - [ ] Build comparative evaluation between different prompt templates and models
- [x] Develop enhanced CLI and API interfaces
  - [x] Update CLI interface to support question answering mode
  - [ ] Create JSON API for programmatic access to question answering
  - [ ] Add interactive mode with follow-up question support
  - [ ] Implement answer streaming for better user experience

### Phase 10: Project Cleanup and Refactoring

- [x] Conduct comprehensive code audit
  - [x] Identify and remove unused imports across the codebase
  - [x] Delete deprecated and unused scripts
  - [x] Clean up orphaned modules not referenced by active CLI commands
  - [x] Archive experimental code with clear documentation
  - [x] Remove redundant utility functions and duplicate code
- [x] Simplify architectural patterns
  - [x] Review dependency injection implementation
  - [x] Consider refactoring to more explicit dependency patterns for better LSP support
  - [x] Simplify complex inheritance hierarchies
  - [x] Consolidate similar services and interfaces
  - [x] Document architectural decisions and patterns
- [x] Streamline CLI interface
  - [x] Consolidate duplicate or overlapping CLI commands
  - [x] Remove obsolete command aliases
  - [x] Group related commands under meaningful subcommands
  - [x] Standardize command parameter naming conventions
  - [x] Update help text for clarity and consistency

### Phase 11: Deployment and Monitoring

#### Component Separation Strategy

Separate the RAG system into two distinct components:

1. **Document Ingestion Pipeline** (build-time)
2. **Question Answering Service** (runtime)

#### Document Ingestion Pipeline Implementation

- [ ] Create dedicated ingestion container

  - [ ] Implement document processing workflow
  - [ ] Design pipeline for batch processing all documentation
  - [ ] Add progress tracking and error handling for large-scale ingestion
  - [ ] Create output validation to ensure data quality
  - [ ] Generate comprehensive pipeline statistics

- [ ] Implement database build process

  - [ ] Configure SQLite document store for production use
  - [ ] Optimize Qdrant vector database settings for deployment
  - [ ] Implement BM25 index generation and serialization
  - [ ] Create database compression and optimization step
  - [ ] Add database integrity verification

- [ ] Design artifact management system
  - [ ] Create versioned storage for document DB snapshots
  - [ ] Implement vector DB export/import functionality
  - [ ] Add BM25 index serialization and loading
  - [ ] Design metadata tracking for database versions
  - [ ] Implement validation for database consistency

#### Question Answering Service Implementation

- [ ] Create lightweight QA container

  - [ ] Implement database loading from artifacts
  - [ ] Design optimized read-only access patterns
  - [ ] Add health checks and readiness probes
  - [ ] Implement performance monitoring hooks
  - [ ] Create scaling guidance and configuration options

- [ ] Develop REST API for question answering

  - [ ] Design query endpoint with parameter validation
  - [ ] Add streaming response capability
  - [ ] Implement rate limiting and request throttling
  - [ ] Create authentication and authorization layer
  - [ ] Add CORS and security headers

- [ ] Implement query performance optimizations
  - [ ] Optimize retrieval for speed in production environment
  - [ ] Add response caching for common queries
  - [ ] Implement query batching for efficiency
  - [ ] Design warm-up procedures for cold starts
  - [ ] Create resource utilization monitoring

#### Deployment Infrastructure

- [ ] Containerize both application components

  - [ ] Create multi-stage Dockerfile for ingestion pipeline
  - [ ] Design slim Dockerfile for question answering service
  - [ ] Implement CI/CD pipeline for container builds
  - [ ] Set up container scanning and security verification
  - [ ] Create deployment configuration templates

- [ ] Set up model and data versioning

  - [ ] Implement artifact registry for database storage
  - [ ] Create versioning strategy for embeddings and models
  - [ ] Add tracking for database schema changes
  - [ ] Design upgrade paths for database migrations
  - [ ] Implement rollback capabilities for failed deployments

- [ ] Implement monitoring for system performance

  - [ ] Set up metrics collection for query performance
  - [ ] Create dashboards for system health visualization
  - [ ] Implement alerting for system degradation
  - [ ] Add detailed logging for troubleshooting
  - [ ] Design SLOs and SLIs for system reliability

- [ ] Create deployment pipeline for continuous updates

  - [ ] Implement blue/green deployment strategy
  - [ ] Design canary releases for controlled rollouts
  - [ ] Create automated testing for deployment verification
  - [ ] Add deployment approval workflows
  - [ ] Implement automated rollback triggers

- [ ] Develop user interface for querying the system
  - [ ] Create simple web UI for direct questions
  - [ ] Add support for query history and favorites
  - [ ] Implement source citation display
  - [ ] Add user feedback collection mechanism
  - [ ] Create admin interface for system management

## Document Overwrite Implementation

To implement document replacement strategy (Option 1):

1. **Update SQLiteDocumentRepository**:

   - Add a `find_document_by_epic_page_id(epic_page_id)` method to locate existing documents
   - Create a `replace_document(document)` method that:
     - Checks if a document with the same epic_page_id exists
     - If exists, deletes the old document and all its chunks
     - Saves the new document with fresh UUIDs
     - Returns the saved document with updated IDs

2. **Modify the IngestDocumentUseCase**:

   - Update to use the new replace_document method
   - Detect if a document should replace an existing one
   - Maintain chunk relationships after replacement

3. **Update Document Processing Pipeline**:

   - Modify the ingest_documents step to use document replacement
   - Add tracking of replaced vs. new documents in statistics
   - Ensure vector store properly handles deleted embeddings

4. **Optimize for Performance**:

   - Add batch operations for document replacement
   - Create transaction support for atomicity
   - Add logging for replacement operations

5. **Add Database Utilities**:
   - Create command to clean orphaned chunks
   - Add database integrity check
   - Implement maintenance operations (vacuum)

## Design Principles

1. **Modularity**: Components should be loosely coupled
2. **Testability**: All core logic should be easily testable
3. **Extensibility**: System should support different embedding models and LLMs
4. **Observability**: Performance metrics should be trackable
5. **Maintainability**: Code should follow clean code practices and include documentation
