"""Evaluation and testing CLI commands."""

import asyncio
import time
from typing import Optional, List

import typer
from rich.markdown import Markdown

from ....infrastructure.container import container
from ....domain.services.contextual_enrichment_service import (
    ContextualEnrichmentService,
)
from ....domain.services.reranker_service import RerankerService
from .common import console, create_progress_bar

evaluation_app = typer.Typer(pretty_exceptions_enable=False)


@evaluation_app.command("test-enrichment")
def test_enrichment(
    markdown_file: str = typer.Argument(
        ..., help="Markdown file to test enrichment on"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for enriched content"
    ),
):
    """Test contextual enrichment on a markdown document."""
    # Import here to avoid circular imports
    from ....domain.services.chunking_service import ChunkingService
    from ....domain.models.document import Document

    # Read the markdown file
    with open(markdown_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create the document
    document = Document(
        title=markdown_file.split("/")[-1].replace(".md", ""),
        content=content,
        source_path=markdown_file,
    )

    # Get the chunking service
    chunking_service = container.resolve(ChunkingService)
    enrichment_service = container.resolve(ContextualEnrichmentService)

    # Chunk the document
    chunking_options = {
        "dynamic_chunking": True,
        "min_chunk_size": 300,
        "max_chunk_size": 800,
        "chunk_overlap": 50,
    }

    console.print(f"[bold]Chunking document:[/bold] {document.title}")
    chunks = chunking_service.chunk_document(document, chunking_options)
    console.print(f"[bold]Generated [cyan]{len(chunks)}[/cyan] chunks[/bold]")

    # Enrich the chunks
    console.print("[bold]Enriching chunks...[/bold]")

    async def enrich_chunks():
        enriched_chunks = []

        with create_progress_bar() as progress:
            task = progress.add_task("Enriching chunks", total=len(chunks), status="")

            for i, chunk in enumerate(chunks):
                progress.update(
                    task, advance=0, status=f"Enriching chunk {i+1}/{len(chunks)}"
                )

                # Enrich the chunk
                enriched_chunk = await enrichment_service.enrich_chunk(chunk)
                enriched_chunks.append(enriched_chunk)

                progress.update(task, advance=1)

        return enriched_chunks

    enriched_chunks = asyncio.run(enrich_chunks())

    # Display the enriched chunks
    for i, chunk in enumerate(enriched_chunks):
        console.print(f"[bold cyan]Chunk {i+1}/{len(enriched_chunks)}[/bold cyan]")
        console.print(f"[bold]Original Content:[/bold]")
        console.print(Markdown(chunk.content))

        console.print(f"[bold]Enrichment:[/bold]")
        if hasattr(chunk, "metadata") and chunk.metadata.get("context_description"):
            console.print(Markdown(chunk.metadata["context_description"]))
        else:
            console.print("[italic]No enrichment available[/italic]")

        console.print()

    # Write to output file if specified
    if output:
        with open(output, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(enriched_chunks):
                f.write(f"## Chunk {i+1}/{len(enriched_chunks)}\n\n")
                f.write("### Original Content\n\n")
                f.write(f"{chunk.content}\n\n")

                f.write("### Enrichment\n\n")
                if hasattr(chunk, "metadata") and chunk.metadata.get(
                    "context_description"
                ):
                    f.write(f"{chunk.metadata['context_description']}\n\n")
                else:
                    f.write("No enrichment available\n\n")

                f.write("---\n\n")

        console.print(f"[bold green]Wrote enriched content to:[/bold green] {output}")


@evaluation_app.command("test-rerank")
def test_rerank(
    query: str = typer.Argument(..., help="Query to test reranking with"),
    passages: List[str] = typer.Argument(..., help="Passages to rerank"),
):
    """Test reranking by reranking passages for a query."""
    # Get the reranker service
    reranker_service = container.resolve(RerankerService)

    console.print(f"[bold]Query:[/bold] {query}")
    console.print()

    # Rerank the passages
    async def rerank_passages():
        scores = await reranker_service.rerank(query, passages)
        return scores

    scores = asyncio.run(rerank_passages())

    # Display the reranked passages
    console.print("[bold]Reranked Passages:[/bold]")
    console.print()

    # Pair passages with scores and sort by score
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

    for i, (passage, score) in enumerate(ranked_passages):
        console.print(f"[bold cyan]{i+1}.[/bold cyan] [bold]Score:[/bold] {score:.4f}")
        console.print(Markdown(passage))
        console.print()


@evaluation_app.command("benchmark-bm25")
def benchmark_bm25(
    query_file: str = typer.Argument(..., help="File containing queries to benchmark"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
):
    """Benchmark BM25 search performance with queries from a file."""
    from ....domain.services.lexical_search_service import LexicalSearchService

    # Read the query file
    with open(query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    # Get the lexical search service
    search_service = container.resolve(LexicalSearchService)

    console.print(f"[bold]Benchmarking {len(queries)} queries...[/bold]")

    # Benchmark the queries
    async def benchmark():
        results = []

        with create_progress_bar() as progress:
            task = progress.add_task(
                "Processing queries", total=len(queries), status=""
            )

            for i, query in enumerate(queries):
                progress.update(task, advance=0, status=f"Query {i+1}/{len(queries)}")

                # Search for the query
                start_time = time.time()
                search_results = await search_service.search(query, top_k)
                elapsed_time = time.time() - start_time

                # Store the results
                results.append(
                    {
                        "query": query,
                        "time": elapsed_time,
                        "result_count": len(search_results),
                    }
                )

                progress.update(task, advance=1)

        return results

    benchmark_results = asyncio.run(benchmark())

    # Calculate statistics
    total_time = sum(result["time"] for result in benchmark_results)
    avg_time = total_time / len(benchmark_results)
    min_time = min(result["time"] for result in benchmark_results)
    max_time = max(result["time"] for result in benchmark_results)

    # Display the results
    console.print("[bold]Benchmark Results:[/bold]")
    console.print(f"Total queries: {len(benchmark_results)}")
    console.print(f"Average time: {avg_time:.4f}s")
    console.print(f"Minimum time: {min_time:.4f}s")
    console.print(f"Maximum time: {max_time:.4f}s")
    console.print(f"Total time: {total_time:.4f}s")


@evaluation_app.command("evaluate-enrichment")
def evaluate_enrichment(
    dataset: str = typer.Option(
        None, "--dataset", "-d", help="JSON file with evaluation dataset"
    ),
    run_generator: bool = typer.Option(
        False, "--generate-dataset", "-g", help="Generate an evaluation dataset"
    ),
    output: str = typer.Option(
        "evaluation_results.json", "--output", "-o", help="Output file for results"
    ),
):
    """Evaluate the effectiveness of contextual enrichment."""
    import json
    from ....application.pipelines.evaluation.dataset_generator import DatasetGenerator
    from ....application.pipelines.evaluation.contextual_enrichment_pipeline import (
        ContextualEnrichmentEvaluationPipeline,
    )

    if run_generator:
        # Generate an evaluation dataset
        console.print("[bold]Generating evaluation dataset...[/bold]")

        # Get the dataset generator
        dataset_generator = container.resolve(DatasetGenerator)

        # Generate the dataset
        evaluation_data = asyncio.run(dataset_generator.generate())

        # Save the dataset
        if not dataset:
            dataset = "evaluation_dataset.json"

        with open(dataset, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2)

        console.print(f"[bold green]Generated dataset saved to:[/bold green] {dataset}")

    # Evaluate the contextual enrichment
    if dataset:
        console.print(
            f"[bold]Evaluating contextual enrichment with dataset:[/bold] {dataset}"
        )

        # Load the dataset
        with open(dataset, "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)

        # Get the evaluation pipeline
        evaluation_pipeline = container.resolve(ContextualEnrichmentEvaluationPipeline)

        # Run the evaluation
        results = asyncio.run(evaluation_pipeline.evaluate(evaluation_data))

        # Save the results
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Display summary
        console.print("[bold]Evaluation Results:[/bold]")
        console.print(f"Mean Reciprocal Rank: {results['metrics']['mrr']:.4f}")
        console.print(f"Average Precision: {results['metrics']['avg_precision']:.4f}")
        console.print(f"[bold]Full results saved to:[/bold] {output}")
    elif not run_generator:
        console.print(
            "[bold red]Error: No dataset specified. Use --dataset or --generate-dataset.[/bold red]"
        )


def register_commands(app: typer.Typer):
    """Register evaluation commands with the main app."""
    app.add_typer(
        evaluation_app, name="evaluate", help="Evaluation and testing commands"
    )
