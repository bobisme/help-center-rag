#!/usr/bin/env python3
"""Demo script for testing image description generation."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

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


async def main():
    """Run the image description demo."""
    parser = argparse.ArgumentParser(
        description="Demo script for image description generation"
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
        "--model",
        type=str,
        default="gemma3:27b",
        help="Ollama model to use for image description",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--doc-path",
        type=str,
        default=None,
        help="Path to a markdown document to extract images from",
    )

    args = parser.parse_args()

    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]Image Description Demo[/bold blue]\n"
            f"Using model: [green]{args.model}[/green]",
            border_style="blue",
        )
    )

    # Create the image description service
    image_service = OllamaImageDescriptionService(
        settings=settings.llm, model=args.model, min_image_size=args.min_size
    )

    # Find images to process
    if args.doc_path:
        # Extract images from document
        with open(args.doc_path, "r") as f:
            document_content = f.read()

        # Fix path issues in document content
        adjusted_content = document_content.replace("![](output/images/", "![](")

        console.print(f"Extracting images from [bold]{args.doc_path}[/bold]...")
        image_data = await image_service.extract_image_contexts(
            adjusted_content, args.image_dir
        )

        # Limit the number of images processed if needed
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

    else:
        # Scan directory for images
        console.print(f"Scanning directory: [bold]{args.image_dir}[/bold]")

        image_files = []
        for root, _, files in os.walk(args.image_dir):
            for file in files:
                if file.lower().endswith((".png", ".gif", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, file))

        # Limit the number of images processed if needed
        if args.sample_limit and len(image_files) > args.sample_limit:
            console.print(
                f"Limiting to {args.sample_limit} images from {len(image_files)} found"
            )
            image_files = image_files[: args.sample_limit]

        if not image_files:
            console.print("[bold red]No images found in directory[/bold red]")
            return

        # Create image data with empty context (directory scan mode)
        image_data = [(img_path, "") for img_path in image_files]
        console.print(f"Found [bold green]{len(image_data)}[/bold green] images")

    # Process images and get descriptions
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Generating image descriptions...", total=len(image_data)
        )

        # Generate descriptions
        results = {}
        for img_path, context in image_data:
            description = await image_service.generate_image_description(
                img_path, context
            )
            if description:  # Some images might be filtered out
                basename = os.path.basename(img_path)
                results[basename] = {"path": img_path, "description": description}
            progress.update(task, advance=1)

    # Display results
    console.print(
        f"\n[bold green]Generated descriptions for {len(results)} images[/bold green]"
    )

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Image")
    table.add_column("Description")

    for img_name, data in results.items():
        table.add_row(f"[cyan]{img_name}[/cyan]", data["description"])

    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
