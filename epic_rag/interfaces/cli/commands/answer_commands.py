"""Question answering CLI commands."""

import asyncio
import time

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from ....infrastructure.container import container
from .common import console

answer_app = typer.Typer(pretty_exceptions_enable=False)


@answer_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="The question to answer"),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of context chunks to retrieve"
    ),
    show_context: bool = typer.Option(
        False, "--show-context", "-c", help="Show the context chunks used"
    ),
    show_metrics: bool = typer.Option(
        False, "--show-metrics", "-m", help="Show processing metrics"
    ),
    temperature: float = typer.Option(
        0.3, "--temperature", "-t", help="Temperature for answer generation"
    ),
    transform_query: bool = typer.Option(
        True, "--transform/--no-transform", help="Transform query using LLM"
    ),
    max_context_chunks: int = typer.Option(
        5, "--max-chunks", help="Maximum number of context chunks to include"
    ),
    min_relevance: float = typer.Option(
        0.5,
        "--min-relevance",
        help="Minimum relevance score for context chunks (lower = more chunks)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output, including the prompt sent to the LLM",
    ),
):
    """Ask a question and get an answer based on the documentation.

    This command uses RAG to search for relevant context and then
    generates an answer based on that context.
    """
    # Get the answer question use case
    answer_use_case = container.get("answer_question_use_case")

    # Print the question
    console.print(f"[bold purple]Question:[/bold purple] {question}")
    console.print()

    # Execute the use case
    start_time = time.time()
    console.print(f"[dim]Retrieving context for: {question}[/dim]")
    result = asyncio.run(
        answer_use_case.execute(
            question=question,
            first_stage_k=top_k * 2,
            second_stage_k=top_k,
            min_relevance_score=min_relevance,
            use_query_transformation=transform_query,
            temperature=temperature,
            max_context_chunks=max_context_chunks,
        )
    )
    elapsed_time = time.time() - start_time

    # Extract the answer and metrics
    answer = result["answer"]
    metrics = result["metrics"]
    context_chunks = result["context_chunks"]

    # Show debug info about context chunks
    console.print(f"[dim]Found {len(context_chunks)} relevant context chunks[/dim]")
    if len(context_chunks) == 0:
        console.print(
            "[yellow]Warning: No context chunks were found. The answer may be generic or incomplete.[/yellow]"
        )

    # Display the answer
    console.print(
        Panel.fit(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )
    console.print()

    # Display verbose info if requested
    if verbose:
        # Simulate what the prompt would look like
        if context_chunks:
            sample_context = "\n...\n".join(
                [
                    f"DOCUMENT TITLE: {chunk.get('title', 'Unknown')}\n{chunk.get('content', '')[:200]}..."
                    for chunk in context_chunks[:2]
                ]
            )
            if len(context_chunks) > 2:
                sample_context += (
                    f"\n...\n[{len(context_chunks) - 2} more chunks not shown]"
                )
        else:
            sample_context = "No relevant information found."

        console.print("[bold]Generated prompt structure:[/bold]")
        console.print("CONTEXT:")
        console.print(f"[dim]{sample_context}[/dim]")
        console.print("\nUSER QUESTION:")
        console.print(f"[dim]{question}[/dim]")
        console.print()

    # Display metrics if requested
    if show_metrics:
        console.print("[bold]Performance Metrics:[/bold]")
        console.print(f"Total time: {metrics['total_time_ms']/1000:.2f}s")
        console.print(f"Retrieval time: {metrics['retrieval_time_ms']/1000:.2f}s")
        console.print(f"Answer generation time: {metrics['answer_time_ms']/1000:.2f}s")
        console.print(f"Chunks retrieved: {metrics['chunks_retrieved']}")
        console.print(f"Chunks used for context: {metrics['chunks_used']}")
        console.print(f"Model: {metrics['model_used']}")
        console.print()

    # Display context chunks if requested
    if show_context:
        console.print("[bold]Context Used:[/bold]")
        console.print()

        for i, chunk in enumerate(context_chunks):
            title = chunk["title"]
            score = chunk["score"]
            console.print(
                f"[bold cyan]{i+1}.[/bold cyan] [bold]{title}[/bold] (Score: {score:.4f})"
            )
            console.print()
            console.print(
                Markdown(
                    chunk["content"][:800] + "..."
                    if len(chunk["content"]) > 800
                    else chunk["content"]
                )
            )
            console.print("---")
            console.print()


def register_commands(app: typer.Typer):
    """Register answer commands with the main app."""
    app.add_typer(answer_app, name="answer", help="Question answering commands")

    # Add the main ask command directly
    app.command(name="ask")(ask)
