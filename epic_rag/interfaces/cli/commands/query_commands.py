"""Query and search-related CLI commands."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict

from ....domain.models.document import DocumentChunk
from ....domain.models.retrieval import Query, RetrievalResult

import typer
from rich.markdown import Markdown

from ....infrastructure.container import container
from .common import console


@dataclass
class SearchResult:
    """Search result from lexical search."""

    id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


query_app = typer.Typer(pretty_exceptions_enable=False)


@query_app.command("query")
def query(
    query_text: str = typer.Argument(..., help="The query text"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    show_metadata: bool = typer.Option(
        False, "--show-metadata", "-m", help="Show document metadata"
    ),
    rerank: bool = typer.Option(
        True, "--rerank/--no-rerank", help="Apply reranking to results"
    ),
    vector_only: bool = typer.Option(
        False, "--vector-only", help="Use only vector search (no BM25 or fusion)"
    ),
    bm25_only: bool = typer.Option(
        False, "--bm25-only", help="Use only BM25 search (no vector search or fusion)"
    ),
    bm25_weight: float = typer.Option(
        0.4, "--bm25-weight", help="Weight for BM25 results in hybrid search"
    ),
    vector_weight: float = typer.Option(
        0.6, "--vector-weight", help="Weight for vector results in hybrid search"
    ),
    full_content: bool = typer.Option(
        True, "--full-content/--preview", help="Show full content or just a preview"
    ),
    transform_query: bool = typer.Option(
        True, "--transform/--no-transform", help="Transform query using LLM"
    ),
    show_details: bool = typer.Option(
        False, "--show-details", "-d", help="Show detailed metrics about search process"
    ),
):
    """Query the system with comprehensive RAG retrieval.

    This command performs a complete search using both semantic (vector) and lexical (BM25)
    search methods, combines the results using rank fusion, and applies reranking
    for improved relevance.
    """
    # Get required services
    search_service = container.get("bm25_search_service")
    llm_service = container.get("llm_service")
    embedding_service = container.get("embedding_service")
    rank_fusion_service = container.get("rank_fusion_service")
    retrieve_use_case = container.get("retrieve_context_use_case")
    reranker_service = container.get("reranker_service") if rerank else None

    # Initialize metrics dictionary
    metrics = {
        "query": query_text,
        "timestamps": {},
        "durations_ms": {},
        "counts": {},
        "weights": {
            "bm25": bm25_weight,
            "vector": vector_weight,
        },
        "options": {
            "rerank": rerank,
            "vector_only": vector_only,
            "bm25_only": bm25_only,
            "transform_query": transform_query,
        },
    }

    # Print query
    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print()

    # Create the overall timer
    overall_start = time.time()
    metrics["timestamps"]["start"] = overall_start

    # We'll move query transformation inside the async function
    async def transform_query_async():
        if transform_query and not (vector_only or bm25_only):
            transform_start = time.time()
            try:
                # Use the dedicated transform_query method from the LLM service
                transformed = await llm_service.transform_query(query_text)
                transform_time = (time.time() - transform_start) * 1000
                
                metrics["transformed_query"] = transformed
                metrics["durations_ms"]["transform"] = transform_time
                
                if show_details:
                    console.print(f"[bold]Transformed Query:[/bold] {transformed}")
                    console.print(f"[dim]Transform time: {transform_time:.2f}ms[/dim]")
                    console.print()
                
                return transformed
            except Exception as e:
                console.print(f"[bold red]Error transforming query:[/bold red] {str(e)}")
                return query_text
        return query_text
        
    # We'll handle the transformation in the main async function

    async def perform_search():
        from ....domain.models.retrieval import Query

        # First transform the query if needed
        transformed_query = await transform_query_async()
        
        # Create a proper Query object
        query_obj = Query(text=transformed_query)

        results = []

        # If vector-only mode
        if vector_only:
            vector_start = time.time()

            # Execute vector search
            retrieval_result = await retrieve_use_case.execute(
                transformed_query,
                first_stage_k=top_k * 2,
                second_stage_k=top_k,
            )

            vector_time = (time.time() - vector_start) * 1000
            metrics["durations_ms"]["vector"] = vector_time
            metrics["counts"]["vector_results"] = len(retrieval_result.final_chunks)

            # Convert to standard result format
            results = [
                SearchResult(
                    id=chunk.id,
                    title=chunk.metadata.get("document_title", "Untitled"),
                    content=chunk.content,
                    score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                    metadata=chunk.metadata,
                )
                for chunk in retrieval_result.final_chunks
            ]

        # If BM25-only mode
        elif bm25_only:
            bm25_start = time.time()

            # Execute BM25 search
            bm25_result = await search_service.search(query_obj, top_k * 2)

            bm25_time = (time.time() - bm25_start) * 1000
            metrics["durations_ms"]["bm25"] = bm25_time
            metrics["counts"]["bm25_results"] = len(bm25_result.chunks)

            # Convert the retrieval result to our standard format
            results = [
                SearchResult(
                    id=chunk.id,
                    title=chunk.metadata.get("document_title", "Untitled"),
                    content=chunk.content,
                    score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                    metadata=chunk.metadata,
                )
                for chunk in bm25_result.chunks
            ]

        # Default hybrid search mode
        else:
            # Run BM25 search
            bm25_start = time.time()
            bm25_result = await search_service.search(query_obj, top_k * 2)
            bm25_time = (time.time() - bm25_start) * 1000

            metrics["durations_ms"]["bm25"] = bm25_time
            metrics["counts"]["bm25_results"] = len(bm25_result.chunks)

            # Convert the retrieval result to our standard format
            bm25_results = [
                SearchResult(
                    id=chunk.id,
                    title=chunk.metadata.get("document_title", "Untitled"),
                    content=chunk.content,
                    score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                    metadata=chunk.metadata,
                )
                for chunk in bm25_result.chunks
            ]

            # Run vector search
            vector_start = time.time()
            retrieval_result = await retrieve_use_case.execute(
                transformed_query,
                first_stage_k=top_k * 2,
                second_stage_k=top_k,
                filter_metadata={"rerank": False},  # Don't rerank yet
            )
            vector_time = (time.time() - vector_start) * 1000

            metrics["durations_ms"]["vector"] = vector_time

            # Convert vector results to standard format
            vector_results = []
            if retrieval_result.final_chunks:
                vector_results = [
                    SearchResult(
                        id=chunk.id,
                        title=chunk.metadata.get("document_title", "Untitled"),
                        content=chunk.content,
                        score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                        metadata=chunk.metadata,
                    )
                    for chunk in retrieval_result.final_chunks
                ]

            metrics["counts"]["vector_results"] = len(vector_results)

            # Combine using rank fusion
            fusion_start = time.time()

            # Check and adapt parameters for the rank fusion service
            # Create a retrieval result from vector search if needed
            vector_retrieval_result = RetrievalResult(
                query_id=retrieval_result.query.id,
                chunks=retrieval_result.final_chunks,
                latency_ms=retrieval_result.total_latency_ms,
            )

            # Now call the rank fusion service with proper parameters
            fused_results = await rank_fusion_service.fuse_results(
                vector_results=vector_retrieval_result,
                bm25_results=bm25_result,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
            )
            fusion_time = (time.time() - fusion_start) * 1000

            metrics["durations_ms"]["fusion"] = fusion_time
            metrics["counts"]["fused_results"] = len(fused_results.chunks)

            # Convert to our standard result format
            results = [
                SearchResult(
                    id=chunk.id,
                    title=chunk.metadata.get("document_title", "Untitled"),
                    content=chunk.content,
                    score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                    metadata=chunk.metadata,
                )
                for chunk in fused_results.chunks
            ]

        # Apply reranking if requested
        if rerank and reranker_service and results:
            rerank_start = time.time()

            # Create a proper Query object
            query = Query(text=query_text)
            
            # Create DocumentChunk objects from our SearchResult objects
            doc_chunks = [
                DocumentChunk(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    relevance_score=result.score
                ) 
                for result in results[: top_k * 2]
            ]

            # Rerank the results
            reranked_chunks = await reranker_service.rerank(query, doc_chunks)
            
            # Map the reranked chunks back to our results
            # Create a mapping from chunk ID to result index
            chunk_id_to_index = {result.id: i for i, result in enumerate(results)}
            
            # Update scores based on the reranked chunks
            for chunk in reranked_chunks:
                if chunk.id in chunk_id_to_index:
                    idx = chunk_id_to_index[chunk.id]
                    results[idx].score = chunk.relevance_score or 0.0
            
            # Sort by the new scores
            results = sorted(results, key=lambda x: x.score, reverse=True)

            rerank_time = (time.time() - rerank_start) * 1000
            metrics["durations_ms"]["rerank"] = rerank_time

        return results[:top_k]

    # Execute the search
    results = asyncio.run(perform_search())

    # Calculate total time
    overall_time = (time.time() - overall_start) * 1000
    metrics["durations_ms"]["total"] = overall_time
    metrics["timestamps"]["end"] = time.time()
    metrics["counts"]["results"] = len(results)

    # Display the results
    if show_details:
        # Show detailed metrics
        console.print(f"[bold]Search Details:[/bold]")
        search_type = (
            "Vector only" if vector_only else ("BM25 only" if bm25_only else "Hybrid")
        )
        console.print(f"Search type: {search_type}")

        # Show timings
        for step, duration in metrics["durations_ms"].items():
            if step != "total":
                console.print(
                    f"{step.capitalize()}: {duration:.2f}ms ({(duration/overall_time)*100:.1f}%)"
                )

        # Show result counts
        for count_type, count in metrics["counts"].items():
            if count_type != "results":
                console.print(f"{count_type}: {count}")

        console.print()

    console.print(f"[bold]Results:[/bold] (took {overall_time/1000:.2f}s)")
    console.print()

    if not results:
        console.print("[italic]No results found.[/italic]")
        return

    for i, result in enumerate(results):
        title = result.title
        score = result.score
        console.print(
            f"[bold cyan]{i+1}.[/bold cyan] [bold]{title}[/bold] (Score: {score:.4f})"
        )

        # Display the document metadata if requested
        if show_metadata and result.metadata:
            console.print("[bold]Metadata:[/bold]")
            for key, value in result.metadata.items():
                console.print(f"  {key}: {value}")

        # Display the document content
        console.print()
        if full_content:
            console.print(Markdown(result.content))
        else:
            # Display a preview of the content
            preview = (
                result.content[:500] + "..."
                if len(result.content) > 500
                else result.content
            )
            console.print(Markdown(preview))
        console.print()


@query_app.command("bm25")
def bm25_search(
    query_text: str = typer.Argument(..., help="The query text"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    show_metadata: bool = typer.Option(
        False, "--show-metadata", "-m", help="Show document metadata"
    ),
    full_doc: bool = typer.Option(
        False, "--full-doc", "-f", help="Show full document content"
    ),
):
    """Search using BM25 lexical search."""
    # Get the lexical search service
    search_service = container.get("bm25_search_service")

    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print()

    # Search for the query
    from ....domain.models.retrieval import Query

    # Create a proper Query object
    query_obj = Query(text=query_text)

    start_time = time.time()
    retrieval_result = asyncio.run(search_service.search(query_obj, top_k))
    elapsed_time = time.time() - start_time

    # Display the results
    console.print(f"[bold]Results:[/bold] (took {elapsed_time:.2f}s)")
    console.print()

    if not retrieval_result.chunks:
        console.print("[italic]No results found.[/italic]")
        return

    # Convert to SearchResult objects
    results = [
        SearchResult(
            id=chunk.id,
            title=chunk.metadata.get("document_title", "Untitled"),
            content=chunk.content,
            score=getattr(chunk, "score", chunk.relevance_score or 0.0),
            metadata=chunk.metadata,
        )
        for chunk in retrieval_result.chunks
    ]

    for i, result in enumerate(results):
        title = result.title or "Untitled"
        score = result.score
        console.print(
            f"[bold cyan]{i+1}.[/bold cyan] [bold]{title}[/bold] (Score: {score:.4f})"
        )

        # Display the document metadata if requested
        if show_metadata and result.metadata:
            console.print("[bold]Metadata:[/bold]")
            for key, value in result.metadata.items():
                console.print(f"  {key}: {value}")

        # Display the document content
        console.print()
        if full_doc:
            console.print(Markdown(result.content))
        else:
            # Display a preview of the content
            preview = (
                result.content[:500] + "..."
                if len(result.content) > 500
                else result.content
            )
            console.print(Markdown(preview))
        console.print()


@query_app.command("hybrid-search")
def hybrid_search(
    query_text: str = typer.Argument(..., help="The query text"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    show_metadata: bool = typer.Option(
        False, "--show-metadata", "-m", help="Show document metadata"
    ),
    rerank: bool = typer.Option(
        True, "--rerank/--no-rerank", help="Apply reranking to results"
    ),
):
    """Perform hybrid search combining BM25 and vector search."""
    # Get the required services
    search_service = container.get("bm25_search_service")
    retrieve_use_case = container.get("retrieve_context_use_case")
    rank_fusion_service = container.get("rank_fusion_service")
    reranker_service = container.get("reranker_service") if rerank else None

    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print()

    async def hybrid_search_async():
        from ....domain.models.retrieval import Query

        # Create a proper Query object
        query_obj = Query(text=query_text)

        # Get BM25 results
        bm25_result = await search_service.search(query_obj, top_k * 2)

        # Convert to SearchResult objects
        bm25_results = [
            SearchResult(
                id=chunk.id,
                title=chunk.metadata.get("document_title", "Untitled"),
                content=chunk.content,
                score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                metadata=chunk.metadata,
            )
            for chunk in bm25_result.chunks
        ]

        # Get vector search results
        retrieval_result = await retrieve_use_case.execute(
            query_text,
            first_stage_k=top_k * 2,
            second_stage_k=top_k,
            filter_metadata={"rerank": False},
        )

        vector_results = []
        if retrieval_result.final_chunks:
            vector_results = [
                SearchResult(
                    id=chunk.id,
                    title=chunk.document_title or "Untitled",
                    content=chunk.content,
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
                for chunk in retrieval_result.final_chunks
            ]

        # Create a retrieval result from vector search if needed
        vector_retrieval_result = RetrievalResult(
            query_id=retrieval_result.query.id,
            chunks=retrieval_result.final_chunks,
            latency_ms=retrieval_result.total_latency_ms,
        )

        # Combine the results using reciprocal rank fusion
        fused_results = await rank_fusion_service.fuse_results(
            vector_results=vector_retrieval_result,
            bm25_results=bm25_result,
            bm25_weight=0.5,
            vector_weight=0.5,
        )

        # Convert the fused results to SearchResult format
        fused_results = [
            SearchResult(
                id=chunk.id,
                title=chunk.metadata.get("document_title", "Untitled"),
                content=chunk.content,
                score=getattr(chunk, "score", chunk.relevance_score or 0.0),
                metadata=chunk.metadata,
            )
            for chunk in fused_results.chunks
        ]

        # Apply reranking if requested
        if rerank and reranker_service:
            # Extract document content for reranking
            doc_texts = [result.content for result in fused_results[: top_k * 2]]

            # Rerank the results
            reranked_scores = await reranker_service.rerank(query_text, doc_texts)

            # Update the scores
            for i, score in enumerate(reranked_scores):
                if i < len(fused_results):
                    fused_results[i].score = score

            # Sort by the new scores
            fused_results = sorted(fused_results, key=lambda x: x.score, reverse=True)

        return fused_results[:top_k]

    # Execute the hybrid search
    start_time = time.time()
    results = asyncio.run(hybrid_search_async())
    elapsed_time = time.time() - start_time

    # Display the results
    console.print(f"[bold]Results:[/bold] (took {elapsed_time:.2f}s)")
    console.print()

    for i, result in enumerate(results):
        title = result.title
        score = result.score
        console.print(
            f"[bold cyan]{i+1}.[/bold cyan] [bold]{title}[/bold] (Score: {score:.4f})"
        )

        # Display the document metadata if requested
        if show_metadata and result.metadata:
            console.print("[bold]Metadata:[/bold]")
            for key, value in result.metadata.items():
                console.print(f"  {key}: {value}")

        # Display the document content
        console.print()
        console.print(Markdown(result.content))
        console.print()


@query_app.command("transform-query")
def transform_query(
    query_text: str = typer.Argument(..., help="The query text"),
):
    """Transform a query using natural language understanding."""

    llm_service = container.get("llm_service")

    console.print(f"[bold]Original Query:[/bold] {query_text}")
    console.print()

    # Use the existing transform_query method from the LLM service
    # which has a better prompt with instructions to be concise
    response = asyncio.run(llm_service.transform_query(query_text))
    
    # Extract just the transformed query, removing any explanations
    # This assumes the first line contains the transformed query
    transformed_query = response.strip().split("\n")[0]
    
    # Display the transformed query
    console.print(f"[bold]Transformed Query:[/bold] {transformed_query}")


def register_commands(app: typer.Typer):
    """Register query commands with the main app."""
    # Register the main query command directly on the app
    app.command(name="query")(query)

    # Add BM25 command directly
    app.command(name="bm25")(bm25_search)

    # Add hybrid search command directly
    app.command(name="hybrid-search")(hybrid_search)

    # Add transform query command directly
    app.command(name="transform-query")(transform_query)

    # Also register all commands under the 'search' namespace for organization
    app.add_typer(query_app, name="search", help="Search and query commands")
