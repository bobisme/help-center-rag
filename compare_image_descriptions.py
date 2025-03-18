#!/usr/bin/env python3
"""Script to compare image descriptions between Gemma and SmolVLM models."""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from epic_rag.infrastructure.config.settings import settings
from epic_rag.infrastructure.llm.ollama_image_description_service import (
    OllamaImageDescriptionService,
)
from epic_rag.infrastructure.llm.smolvlm_image_description_service import (
    SmolVLMImageDescriptionService,
)


async def main():
    """Run the image description comparison."""
    parser = argparse.ArgumentParser(
        description="Compare image descriptions from different models"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=settings.images_dir,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=64,
        help="Minimum size in pixels for images to process",
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default="gemma3:27b",
        help="Ollama model to use for image description",
    )
    parser.add_argument(
        "--smolvlm-model",
        type=str,
        default="HuggingFaceTB/SmolVLM-Synthetic",
        help="HuggingFace model to use for image description",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--doc-path",
        type=str,
        default="test/samples/renew-a-certificate.md",
        help="Path to a markdown document to extract images from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for SmolVLM (cuda, cpu, mps). If not provided, will auto-detect.",
    )

    args = parser.parse_args()

    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]Image Description Model Comparison[/bold blue]\n"
            f"Comparing [green]{args.gemma_model}[/green] vs [green]{args.smolvlm_model}[/green]",
            border_style="blue",
        )
    )

    # Create the image description services
    gemma_service = OllamaImageDescriptionService(
        settings=settings.llm, model=args.gemma_model, min_image_size=args.min_size
    )

    smolvlm_service = SmolVLMImageDescriptionService(
        settings=settings.llm,
        model_name=args.smolvlm_model,
        min_image_size=args.min_size,
        device=args.device,
    )

    # Extract images from document
    with open(args.doc_path, "r") as f:
        document_content = f.read()

    # Fix path issues in document content
    adjusted_content = document_content.replace("![](output/images/", "![](")

    console.print(f"Extracting images from [bold]{args.doc_path}[/bold]...")
    image_data = await gemma_service.extract_image_contexts(
        adjusted_content, args.image_dir
    )

    # Limit the number of images processed
    if args.sample_limit and len(image_data) > args.sample_limit:
        console.print(
            f"Limiting to {args.sample_limit} images from {len(image_data)} found"
        )
        image_data = image_data[: args.sample_limit]

    if not image_data:
        console.print("[bold red]No images found in document[/bold red]")
        return

    console.print(
        f"Found [bold green]{len(image_data)}[/bold green] images in document"
    )

    # Process images with both models
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        # Gemma task
        gemma_task = progress.add_task(
            "Generating descriptions with Gemma...", total=len(image_data)
        )

        # Generate Gemma descriptions
        gemma_results = {}
        for img_path, context in image_data:
            description = await gemma_service.generate_image_description(
                img_path, context
            )
            if description:
                basename = os.path.basename(img_path)
                gemma_results[basename] = {
                    "path": img_path,
                    "description": description,
                    "context": context[:100] + "..." if len(context) > 100 else context,
                }
            progress.update(gemma_task, advance=1)

        # SmolVLM task
        smolvlm_task = progress.add_task(
            "Generating descriptions with SmolVLM...", total=len(image_data)
        )

        # Generate SmolVLM descriptions
        smolvlm_results = {}
        for img_path, context in image_data:
            description = await smolvlm_service.generate_image_description(
                img_path, context
            )
            if description:
                basename = os.path.basename(img_path)
                smolvlm_results[basename] = {
                    "path": img_path,
                    "description": description,
                }
            progress.update(smolvlm_task, advance=1)

    # Display comparison results
    console.print(
        f"\n[bold green]Generated descriptions for {len(gemma_results)} images[/bold green]"
    )

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Image")
    table.add_column("Surrounding Context")
    table.add_column("Gemma Description")
    table.add_column("SmolVLM Description")

    for img_name, data in gemma_results.items():
        smolvlm_desc = smolvlm_results.get(img_name, {}).get(
            "description", "No description generated"
        )
        table.add_row(
            f"[cyan]{img_name}[/cyan]",
            data["context"],
            data["description"],
            smolvlm_desc,
        )

    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
