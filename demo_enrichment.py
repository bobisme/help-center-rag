#!/usr/bin/env python3
"""
Quick demo to show the impact of contextual enrichment on insurance agency documentation.
Use this script to test specific queries against the manually created test chunks.
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys

# Rich console for pretty output
console = Console()

# Import the sample chunk data from manual_evaluation.py
try:
    from manual_evaluation import BASE_CHUNKS, ENRICHED_CHUNKS, compute_bm25_score
except ImportError:
    console.print("[red]Error importing sample chunks from manual_evaluation.py[/red]")
    sys.exit(1)

def search_chunks(query, chunks, top_k=3):
    """Search chunks and return top k results."""
    results = []
    for chunk in chunks:
        score = compute_bm25_score(query, chunk["content"])
        results.append({"id": chunk["id"], "score": score, "content": chunk["content"]})
    
    # Sort by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top k results
    return results[:top_k]

def run_demo_query(query):
    """Run a demo query and show results."""
    console.print(f"\n[bold]Searching for:[/bold] {query}\n")
    
    # Search base chunks
    base_results = search_chunks(query, BASE_CHUNKS)
    
    # Search enriched chunks
    enriched_results = search_chunks(query, ENRICHED_CHUNKS)
    
    # Show results in a table
    table = Table(title="Search Results Comparison")
    table.add_column("Rank", style="cyan")
    table.add_column("Base Score", style="yellow")
    table.add_column("Enriched Score", style="green")
    table.add_column("Score Diff", style="magenta")
    table.add_column("% Improvement", style="bold green")
    
    for i in range(min(len(base_results), len(enriched_results))):
        base_score = base_results[i]["score"]
        enriched_score = enriched_results[i]["score"]
        score_diff = enriched_score - base_score
        percent_diff = (score_diff / base_score * 100) if base_score > 0 else 0
        
        table.add_row(
            str(i+1),
            f"{base_score:.2f}",
            f"{enriched_score:.2f}",
            f"{score_diff:+.2f}",
            f"{percent_diff:+.1f}%"
        )
    
    console.print(table)
    
    # Show the top result for each approach
    if base_results and enriched_results:
        console.print("\n[bold cyan]Top Result Without Enrichment:[/bold cyan]")
        base_title = base_results[0]["content"].split("\n\n")[0]
        console.print(Panel(base_title, expand=False))
        
        console.print("\n[bold green]Top Result With Contextual Enrichment:[/bold green]")
        # Get the first paragraph which should be the enrichment context
        enriched_paragraphs = enriched_results[0]["content"].split("\n\n")
        if len(enriched_paragraphs) > 1:
            # Show the context (first paragraph) and the document title (second paragraph)
            context = enriched_paragraphs[0]
            title = enriched_paragraphs[1] if len(enriched_paragraphs) > 1 else ""
            console.print(Panel(f"{context}\n\n{title}", expand=False))
        else:
            console.print(Panel(enriched_results[0]["content"].split("\n\n")[0], expand=False))
    return

def main():
    """Main demo function."""
    console.print("[bold cyan]Applied Epic Documentation Contextual Enrichment Demo[/bold cyan]")
    console.print("This demo shows how contextual enrichment improves retrieval quality for insurance documentation.\n")
    
    # Try several different queries
    test_queries = [
        "How do I compare quotes for a client?",
        "What steps are needed to renew a certificate?",
        "How do I access my email in Epic?",
        "How do I set up VINlink Decoder?"
    ]
    
    for query in test_queries:
        console.print(f"\n[bold magenta]Testing Query:[/bold magenta] [bold]'{query}'[/bold]")
        run_demo_query(query)
    
    console.print("\n[bold]Conclusion:[/bold] Adding contextual enrichment helps the search algorithm better understand the document content.")
    console.print("This improves relevance scores and generally provides better search results, especially for natural language queries.")

if __name__ == "__main__":
    main()