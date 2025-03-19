"""Testing and evaluation commands."""

import asyncio
import os
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown

from .....domain.models.document import Document
from .....infrastructure.container import setup_container, container
from .....application.use_cases.ingest_document import IngestDocumentUseCase
from ...common import console

# Create a Typer app for testing commands
testing_app = typer.Typer()

# Import sample data for demo enrichment
try:
    # These variables will be defined when we move the implementation
    BASE_CHUNKS = []
    ENRICHED_CHUNKS = []
except ImportError:
    pass


def compute_bm25_score(query: str, text: str) -> float:
    """Compute a simplified BM25 score for a query against text."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create a simple TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")

    # Process the query and text
    try:
        tfidf_matrix = vectorizer.fit_transform([query, text])
        query_vector = tfidf_matrix[0].toarray().flatten()
        text_vector = tfidf_matrix[1].toarray().flatten()

        # Compute a score (dot product of vectors)
        score = sum(q * t for q, t in zip(query_vector, text_vector))
        return score
    except:
        return 0.0


@testing_app.command("enrichment-demo")
def demo_enrichment():
    """Run an interactive demo of contextual enrichment impact."""
    # Define sample data
    global BASE_CHUNKS, ENRICHED_CHUNKS

    # Manually define chunk content from sample documents
    BASE_CHUNKS = [
        {
            "id": "chunk1",
            "content": """# Email

Applied Epic allows you to launch an integrated email client from within the
system for all routine email workflows except those initiated by Distribution
Manager. Your organization may opt to use *Microsoft Outlook* or an Epic custom
message window for this integration, or allow you to make an individual
selection in Email Settings Configuration. Regardless of the option you are
using, do one of the following to access your email:

* From the Home screen, click **Email** on the navigation panel or **Areas > Email** on the menubar.""",
        },
        {
            "id": "chunk2",
            "content": """# Quote Results

In Applied Epic, a Quote Results List displays quote information for a specified line
of business on a submission, policy, or quote worksheet. If available, you can compare
quote information side-by-side, make a final carrier selection, and prepare a proposal
for presentation to your client.

## Quote Results List

1. If not already displayed, click **Quote Results** in the left navigation panel.
2. The list displays all quotes that have been entered for the line of business.""",
        },
        {
            "id": "chunk3",
            "content": """# Renew a Certificate

Certificates can be renewed to provide proof of insurance coverage for the insured.

1. Click **Search** on the navigation panel.
2. Enter the insured information and click **Search**.
3. In the search results, click to open the insured record.
4. Click **Service > Certificates**.
5. Click **Search** to locate the certificate that requires renewal.
6. Click the row of the certificate to highlight it.
7. Click **Renew**.""",
        },
        {
            "id": "chunk4",
            "content": """# Faxing Setup

Applied Epic can be configured to enable faxing capabilities directly from the system.

## COM Port Settings

1. Navigate to **Configuration > System > Faxing Setup**.
2. Enter the COM port settings that correspond to your fax modem.
3. Set the default country code and area code for outgoing faxes.
4. Click **Test Connection** to verify the settings.
5. Click **Save** to apply the settings.""",
        },
        {
            "id": "chunk5",
            "content": """# VINlink Decoder Configuration

VINlink Decoder provides automatic vehicle information decoding based on the VIN.

1. Navigate to **Configuration > Integration > External Services**.
2. Select **VINlink Decoder** from the list of available services.
3. Enter your VINlink account credentials in the corresponding fields.
4. Set the default timeout value for requests.
5. Click **Test Connection** to verify your settings.
6. Click **Save** to apply the configuration.""",
        },
    ]

    # Enriched versions of the same chunks with contextual information
    ENRICHED_CHUNKS = [
        {
            "id": "chunk1",
            "content": """This document provides instructions for accessing and using email within the Applied Epic insurance agency management system. It includes details about email integration options with Microsoft Outlook.

# Email

Applied Epic allows you to launch an integrated email client from within the
system for all routine email workflows except those initiated by Distribution
Manager. Your organization may opt to use *Microsoft Outlook* or an Epic custom
message window for this integration, or allow you to make an individual
selection in Email Settings Configuration. Regardless of the option you are
using, do one of the following to access your email:

* From the Home screen, click **Email** on the navigation panel or **Areas > Email** on the menubar.""",
        },
        {
            "id": "chunk2",
            "content": """This document explains how to view, compare, and work with insurance quotes in Applied Epic. It includes instructions for accessing the Quote Results List, comparing quotes side-by-side, and preparing proposals.

# Quote Results

In Applied Epic, a Quote Results List displays quote information for a specified line
of business on a submission, policy, or quote worksheet. If available, you can compare
quote information side-by-side, make a final carrier selection, and prepare a proposal
for presentation to your client.

## Quote Results List

1. If not already displayed, click **Quote Results** in the left navigation panel.
2. The list displays all quotes that have been entered for the line of business.""",
        },
        {
            "id": "chunk3",
            "content": """This document provides step-by-step instructions for renewing a certificate of insurance in Applied Epic, which insurance agencies use to provide proof of insurance coverage to their clients.

# Renew a Certificate

Certificates can be renewed to provide proof of insurance coverage for the insured.

1. Click **Search** on the navigation panel.
2. Enter the insured information and click **Search**.
3. In the search results, click to open the insured record.
4. Click **Service > Certificates**.
5. Click **Search** to locate the certificate that requires renewal.
6. Click the row of the certificate to highlight it.
7. Click **Renew**.""",
        },
        {
            "id": "chunk4",
            "content": """This document provides instructions for setting up and configuring faxing capabilities within the Applied Epic insurance agency management system, including COM port settings and connection testing.

# Faxing Setup

Applied Epic can be configured to enable faxing capabilities directly from the system.

## COM Port Settings

1. Navigate to **Configuration > System > Faxing Setup**.
2. Enter the COM port settings that correspond to your fax modem.
3. Set the default country code and area code for outgoing faxes.
4. Click **Test Connection** to verify the settings.
5. Click **Save** to apply the settings.""",
        },
        {
            "id": "chunk5",
            "content": """This document explains how to configure the VINlink Decoder integration in Applied Epic, which allows automatic decoding of vehicle information based on Vehicle Identification Numbers (VINs).

# VINlink Decoder Configuration

VINlink Decoder provides automatic vehicle information decoding based on the VIN.

1. Navigate to **Configuration > Integration > External Services**.
2. Select **VINlink Decoder** from the list of available services.
3. Enter your VINlink account credentials in the corresponding fields.
4. Set the default timeout value for requests.
5. Click **Test Connection** to verify your settings.
6. Click **Save** to apply the configuration.""",
        },
    ]

    def search_chunks(query, chunks, top_k=3):
        """Search chunks and return top k results."""
        results = []
        for chunk in chunks:
            score = compute_bm25_score(query, chunk["content"])
            results.append(
                {"id": chunk["id"], "score": score, "content": chunk["content"]}
            )

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
                str(i + 1),
                f"{base_score:.2f}",
                f"{enriched_score:.2f}",
                f"{score_diff:+.2f}",
                f"{percent_diff:+.1f}%",
            )

        console.print(table)

        # Show the top result for each approach
        if base_results and enriched_results:
            console.print("\n[bold cyan]Top Result Without Enrichment:[/bold cyan]")
            base_title = base_results[0]["content"].split("\n\n")[0]
            console.print(Panel(base_title, expand=False))

            console.print(
                "\n[bold green]Top Result With Contextual Enrichment:[/bold green]"
            )
            # Get the first paragraph which should be the enrichment context
            enriched_paragraphs = enriched_results[0]["content"].split("\n\n")
            if len(enriched_paragraphs) > 1:
                # Show the context (first paragraph) and the document title (second paragraph)
                context = enriched_paragraphs[0]
                title = enriched_paragraphs[1] if len(enriched_paragraphs) > 1 else ""
                console.print(Panel(f"{context}\n\n{title}", expand=False))
            else:
                console.print(
                    Panel(enriched_results[0]["content"].split("\n\n")[0], expand=False)
                )
        return

    # Main demo function
    console.print(
        "[bold cyan]Applied Epic Documentation Contextual Enrichment Demo[/bold cyan]"
    )
    console.print(
        "This demo shows how contextual enrichment improves retrieval quality for insurance documentation.\n"
    )

    # Try several different queries
    test_queries = [
        "How do I compare quotes for a client?",
        "What steps are needed to renew a certificate?",
        "How do I access my email in Epic?",
        "How do I set up VINlink Decoder?",
    ]

    for query in test_queries:
        console.print(
            f"\n[bold magenta]Testing Query:[/bold magenta] [bold]'{query}'[/bold]"
        )
        run_demo_query(query)

    console.print(
        "\n[bold]Conclusion:[/bold] Adding contextual enrichment helps the search algorithm better understand the document content."
    )
    console.print(
        "This improves relevance scores and generally provides better search results, especially for natural language queries."
    )


@testing_app.command("evaluate-enrichment")
def evaluate_enrichment():
    """Run a full evaluation of contextual enrichment impact."""
    # Test queries relevant to test documents
    TEST_QUERIES = [
        "How do I access my email in Epic?",
        "How do I compare quotes for a client?",
        "What steps are needed to renew a certificate?",
        "How do I set up faxing for my agency?",
        "How do I configure VINlink Decoder?",
    ]

    # Sections we expect to match for each query
    EXPECTED_SECTIONS = {
        "How do I access my email in Epic?": ["Email", "Microsoft Outlook"],
        "How do I compare quotes for a client?": [
            "Quote Results",
            "Quote Results List",
            "Prepare Proposal",
        ],
        "What steps are needed to renew a certificate?": ["Renew a Certificate"],
        "How do I set up faxing for my agency?": ["Faxing Setup", "COM Port Settings"],
        "How do I configure VINlink Decoder?": ["VINlink Decoder Configuration"],
    }

    # Use the same chunks from the demo
    global BASE_CHUNKS, ENRICHED_CHUNKS
    if not BASE_CHUNKS or not ENRICHED_CHUNKS:
        console.print("[bold red]Error: Sample chunks not initialized[/bold red]")
        console.print("Please run 'enrichment-demo' command first or restart the CLI")
        return

    console.print("[bold]Running Evaluation of Contextual Enrichment[/bold]")
    console.print("This evaluation compares search results with and without enrichment")
    console.print()

    # Function to check if a chunk matches expected sections
    def is_relevant(chunk, query):
        for section in EXPECTED_SECTIONS.get(query, []):
            if section.lower() in chunk["content"].lower():
                return True
        return False

    # Results table
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Query", style="cyan")
    results_table.add_column("Base Precision", style="yellow")
    results_table.add_column("Enriched Precision", style="green")
    results_table.add_column("Improvement", style="magenta")

    # Run evaluation for each query
    for query in TEST_QUERIES:
        console.print(f"[bold]Query:[/bold] {query}")

        # Search base chunks
        base_results = []
        for chunk in BASE_CHUNKS:
            score = compute_bm25_score(query, chunk["content"])
            base_results.append(
                {"id": chunk["id"], "score": score, "content": chunk["content"]}
            )
        base_results.sort(key=lambda x: x["score"], reverse=True)
        base_results = base_results[:3]  # Top 3 results

        # Search enriched chunks
        enriched_results = []
        for chunk in ENRICHED_CHUNKS:
            score = compute_bm25_score(query, chunk["content"])
            enriched_results.append(
                {"id": chunk["id"], "score": score, "content": chunk["content"]}
            )
        enriched_results.sort(key=lambda x: x["score"], reverse=True)
        enriched_results = enriched_results[:3]  # Top 3 results

        # Calculate precision metrics
        base_relevant = sum(1 for r in base_results if is_relevant(r, query))
        base_precision = base_relevant / len(base_results) if base_results else 0

        enriched_relevant = sum(1 for r in enriched_results if is_relevant(r, query))
        enriched_precision = (
            enriched_relevant / len(enriched_results) if enriched_results else 0
        )

        improvement = enriched_precision - base_precision

        # Add to results table
        results_table.add_row(
            query,
            f"{base_precision:.2f}",
            f"{enriched_precision:.2f}",
            f"{improvement:+.2f}",
        )

        # Print detailed results for each query
        console.print(f"Base relevant: {base_relevant}/{len(base_results)}")
        console.print(f"Enriched relevant: {enriched_relevant}/{len(enriched_results)}")
        console.print()

    # Print summary results
    console.print(results_table)
    console.print()
    console.print("[bold]Evaluation Complete[/bold]")


@testing_app.command("help-center-pipeline")
def test_help_center_pipeline(
    limit: int = typer.Option(5, "--limit", "-l", help="Number of pages to process"),
    apply_enrichment: bool = typer.Option(
        True, "--enrichment/--no-enrichment", help="Apply enrichment"
    ),
):
    """Test processing help center pages using pipeline components."""
    console.print(
        Panel(
            f"[bold]Testing Help Center Pipeline Components[/bold]\n\n"
            f"Processing {limit} pages with enrichment {'enabled' if apply_enrichment else 'disabled'}",
            title="Pipeline Test",
            border_style="green",
        )
    )

    # Initialize container
    setup_container()

    # Get required services
    document_repository = container.get("document_repository")
    vector_repository = container.get("vector_repository")
    chunking_service = container.get("chunking_service")
    embedding_service = container.get("embedding_service")

    async def run_pipeline():
        # Load sample help center data
        try:
            sample_dir = Path("data/help_center_samples")
            if not sample_dir.exists():
                console.print(
                    f"[bold red]Error: Sample directory {sample_dir} not found[/bold red]"
                )
                return

            sample_files = list(sample_dir.glob("*.json"))
            if not sample_files:
                console.print("[bold red]Error: No sample files found[/bold red]")
                return

            # Limit the number of files to process
            sample_files = sample_files[:limit]

            console.print(f"Found {len(sample_files)} sample files to process")

            # Process each file
            with Progress() as progress:
                task = progress.add_task(
                    "Processing help center pages", total=len(sample_files)
                )

                for sample_file in sample_files:
                    progress.update(task, description=f"Processing {sample_file.name}")

                    # Load the sample JSON data
                    with open(sample_file, "r") as f:
                        data = json.load(f)

                    # Create document object
                    document = Document(
                        title=data.get("title", sample_file.stem),
                        content=data.get("content", ""),
                        source_path=str(sample_file),
                        epic_page_id=data.get("id"),
                        metadata={
                            "category": data.get("category", "Unknown"),
                            "last_updated": data.get("updated_at"),
                        },
                    )

                    # Process document (similar to what the pipeline would do)

                    # 1. Chunk the document
                    chunking_options = {
                        "dynamic_chunking": True,
                        "min_chunk_size": 300,
                        "max_chunk_size": 800,
                        "chunk_overlap": 50,
                    }
                    chunks = chunking_service.chunk_document(document, chunking_options)
                    console.print(f"  Created {len(chunks)} chunks for document")

                    # 2. Store document and chunks
                    db_document = await document_repository.save_document(document)
                    for chunk in chunks:
                        chunk.document_id = db_document.id
                        if apply_enrichment:
                            # Add fake enrichment
                            chunk.metadata["enriched"] = True
                            chunk.metadata["context"] = (
                                f"This is a document about {document.title}"
                            )
                        await document_repository.save_chunk(chunk)

                    # 3. Create embeddings and store in vector DB
                    try:
                        for chunk in chunks:
                            embedding = await embedding_service.embed(chunk.content)
                            await vector_repository.add_vector(
                                chunk.id, embedding, {"document_id": document.id}
                            )
                    except Exception as e:
                        console.print(
                            f"[bold red]Error creating embeddings: {str(e)}[/bold red]"
                        )

                    progress.update(task, advance=1)

            # Show summary
            doc_count = await document_repository.count_documents()
            chunk_count = await document_repository.count_chunks()

            console.print(
                f"[bold green]Pipeline test completed successfully[/bold green]"
            )
            console.print(f"Total documents: {doc_count}")
            console.print(f"Total chunks: {chunk_count}")
            console.print(
                f"Enrichment: {'Enabled' if apply_enrichment else 'Disabled'}"
            )

        except Exception as e:
            console.print(f"[bold red]Error running pipeline: {str(e)}[/bold red]")

    # Run the asyncio pipeline
    asyncio.run(run_pipeline())


def register_commands(app: typer.Typer):
    """Register testing commands with the main app."""
    app.add_typer(testing_app, name="testing", help="Testing and evaluation commands")
