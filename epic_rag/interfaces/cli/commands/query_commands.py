"""Query and search-related CLI commands."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict

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
):
    """Query the system with RAG retrieval."""
    # Call the retrieve context use case
    retrieve_use_case = container.get("retrieve_context_use_case")

    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print()

    # Retrieve the context
    start_time = time.time()
    # The rerank parameter might be handled differently in the use case
    # Pass it through filter_metadata if the execute method doesn't support
    # rerank directly
    result = asyncio.run(
        retrieve_use_case.execute(
            query_text,
            first_stage_k=top_k * 2,
            second_stage_k=top_k,
            filter_metadata={"rerank": rerank} if rerank else None,
        )
    )
    elapsed_time = time.time() - start_time

    # Display the context
    console.print(f"[bold]Results:[/bold] (took {elapsed_time:.2f}s)")
    console.print()

    if not result.final_chunks:
        console.print("[italic]No results found.[/italic]")
        return

    for i, chunk in enumerate(result.final_chunks):
        title = chunk.document_title or "Untitled"
        score = chunk.score
        console.print(
            f"[bold cyan]{i+1}.[/bold cyan] [bold]{title}[/bold] (Score: {score:.4f})"
        )

        # Display the document metadata if requested
        if show_metadata and chunk.metadata:
            console.print("[bold]Metadata:[/bold]")
            for key, value in chunk.metadata.items():
                console.print(f"  {key}: {value}")

        # Display the document content
        console.print()
        console.print(Markdown(chunk.content))
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
    start_time = time.time()
    results = asyncio.run(search_service.search(query_text, top_k))
    elapsed_time = time.time() - start_time

    # Display the results
    console.print(f"[bold]Results:[/bold] (took {elapsed_time:.2f}s)")
    console.print()

    for i, result in enumerate(results):
        title = result.titel or "Untitled"
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
        # Get BM25 results
        bm25_results = await search_service.search(query_text, top_k * 2)

        # Get vector search results
        retrieval_result = await retrieve_use_case.execute(
            query_text, top_k * 2, rerank=False
        )
        vector_results = [
            SearchResult(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                score=doc.score,
                metadata=doc.metadata,
            )
            for doc in retrieval_result.documents
        ]

        # Combine the results using reciprocal rank fusion
        fused_results = rank_fusion_service.fuse_results(
            [bm25_results, vector_results], [0.5, 0.5]
        )

        # Apply reranking if requested
        if rerank and reranker_service:
            # Extract document content for reranking
            doc_texts = [doc.content for doc in fused_results[: top_k * 2]]

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

    # Define the prompt for query transformation
    prompt = f"""
You are a query transformation system for a help desk knowledge base.
Your task is to transform the user's query into a more effective search query.
Consider what the user's intent might be and create a query that will be more
likely to retrieve relevant information.

Original Query: {query_text}

Transformed Query:
"""

    # Transform the query
    response = asyncio.run(llm_service.generate(prompt, max_tokens=100))

    # Display the transformed query
    console.print(f"[bold]Transformed Query:[/bold] {response.strip()}")


def register_commands(app: typer.Typer):
    """Register query commands with the main app."""
    app.add_typer(query_app, name="search", help="Search and query commands")
