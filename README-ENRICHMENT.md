# Contextual Enrichment in Epic RAG System

This document explains how the contextual enrichment system works in our Epic RAG application and how to verify it's working correctly.

## What is Contextual Enrichment?

Contextual enrichment is a technique that uses an LLM to generate a short context description for each document chunk before embedding it. This context description helps:

1. Improve retrieval quality by adding semantic information
2. Provide better understanding of where the chunk fits in the overall document
3. Enhance the embedding with additional contextual information

## How it Works

When a document is ingested, the following steps happen:

1. The document is chunked using the chunking service
2. For each chunk, the contextual enrichment service generates a context description
3. The context is added to both:
   - The chunk metadata (as `metadata.context`)
   - Prepended to the chunk content
4. The enriched chunks are embedded and stored in the vector database

## Verifying Enrichment

To verify that enrichment is working correctly, you can:

1. Examine chunk metadata with the `--metadata` flag:
   ```
   ./rag show-doc-chunks "Email" --metadata
   ```

2. Look for two key indicators of enrichment:
   - The `enriched: True` field in metadata
   - The `context: "..."` field containing the generated description

3. Check the chunk content to see if it begins with the context description followed by the original content

## Common Issues

If enrichment isn't being applied, check the following:

1. Make sure the `apply_enrichment` parameter is set to `True` when calling `ingest_use_case.execute()` (this is the default)
2. Verify the `contextual_enrichment_service` is properly injected into the `IngestDocumentUseCase`
3. Check that the LLM service is properly configured and working

## Fixing Missing Enrichment

If you find documents that haven't been enriched, you can run the fix script:

```
python fix_enrichment.py
```

This will scan the database for unenriched chunks and apply enrichment to them.

For a specific document, you can use:

```
python fix_by_sql.py enrich --id document-id-here
```

## Implementation Details

The contextual enrichment happens in two key places:

1. **IngestDocumentUseCase**: In the `execute()` method, there's a step to apply enrichment between chunking and embedding
2. **OllamaContextualEnrichmentService**: Implements the enrichment by generating context and updating chunks

The key code paths:

```python
# During ingestion in IngestDocumentUseCase.execute()
if apply_enrichment and self.contextual_enrichment_service:
    chunks = await self.contextual_enrichment_service.enrich_chunks(
        document=document, 
        chunks=chunks
    )

# In OllamaContextualEnrichmentService.enrich_chunk()
enriched_chunk = DocumentChunk(
    id=chunk.id,
    document_id=chunk.document_id,
    content=f"{context}\n\n{chunk.content}",  # Prepend context to content
    metadata={**chunk.metadata, "enriched": True, "context": context},  # Add to metadata
    embedding=chunk.embedding,
    chunk_index=chunk.chunk_index,
    previous_chunk_id=chunk.previous_chunk_id,
    next_chunk_id=chunk.next_chunk_id,
    relevance_score=chunk.relevance_score,
)
```