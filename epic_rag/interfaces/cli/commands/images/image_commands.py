"""Image description and processing commands."""

import asyncio
import os
from pathlib import Path
from typing import Optional, List, Tuple

import typer
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
from rich.markdown import Markdown

from .....infrastructure.config.settings import settings
from .....infrastructure.llm.ollama_image_description_service import (
    OllamaImageDescriptionService,
)
from .....infrastructure.llm.smolvlm_image_description_service import (
    SmolVLMImageDescriptionService,
)
from ...common import console

# Create Typer app for image commands
image_app = typer.Typer()


@image_app.command("describe")
def describe_images(
    image_dir: str = typer.Option(
        settings.images_dir,
        "--image-dir",
        help="Directory containing images to process",
    ),
    min_size: int = typer.Option(
        64, "--min-size", help="Minimum size in pixels for images to process"
    ),
    model: str = typer.Option(
        "gemma3:27b", "--model", help="Ollama model to use for image description"
    ),
    sample_limit: int = typer.Option(
        5, "--sample-limit", help="Maximum number of images to process"
    ),
    doc_path: Optional[str] = typer.Option(
        None, "--doc-path", help="Path to a markdown document to extract images from"
    ),
):
    """Generate descriptions for images using an LLM model."""
    console.print(
        Panel.fit(
            "[bold blue]Image Description[/bold blue]\n"
            f"Using model: [green]{model}[/green]",
            border_style="blue",
        )
    )

    # Function to process images using LLM
    async def process_images():
        # Create the image description service
        image_service = OllamaImageDescriptionService(
            settings=settings.llm, model=model, min_image_size=min_size
        )

        # Find images to process
        if doc_path:
            # Extract images from document
            with open(doc_path, "r") as f:
                document_content = f.read()

            # Fix path issues in document content
            adjusted_content = document_content.replace("![](output/images/", "![](")

            console.print(f"Extracting images from [bold]{doc_path}[/bold]...")
            image_data = await image_service.extract_image_contexts(
                adjusted_content, image_dir
            )

            # Limit the number of images processed if needed
            if sample_limit and len(image_data) > sample_limit:
                console.print(
                    f"Limiting to {sample_limit} images from {len(image_data)} found"
                )
                image_data = image_data[:sample_limit]

            if not image_data:
                console.print("[bold red]No images found in document[/bold red]")
                return {}

            console.print(
                f"Found [bold green]{len(image_data)}[/bold green] images in document"
            )

        else:
            # Scan directory for images
            console.print(f"Scanning directory: [bold]{image_dir}[/bold]")

            image_files = []
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith((".png", ".gif", ".jpg", ".jpeg")):
                        image_files.append(os.path.join(root, file))

            # Limit the number of images processed if needed
            if sample_limit and len(image_files) > sample_limit:
                console.print(
                    f"Limiting to {sample_limit} images from {len(image_files)} found"
                )
                image_files = image_files[:sample_limit]

            if not image_files:
                console.print("[bold red]No images found in directory[/bold red]")
                return {}

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

        return results

    # Run the async function
    results = asyncio.run(process_images())

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


@image_app.command("describe-smolvlm")
def describe_images_smolvlm(
    image_dir: str = typer.Option(
        settings.images_dir,
        "--image-dir",
        help="Directory containing images to process",
    ),
    min_size: int = typer.Option(
        64, "--min-size", help="Minimum size in pixels for images to process"
    ),
    model: str = typer.Option(
        "HuggingFaceTB/SmolVLM-Synthetic",
        "--model",
        help="HuggingFace model to use for image description",
    ),
    sample_limit: int = typer.Option(
        5, "--sample-limit", help="Maximum number of images to process"
    ),
    doc_path: Optional[str] = typer.Option(
        None, "--doc-path", help="Path to a markdown document to extract images from"
    ),
):
    """Generate descriptions for images using SmolVLM model."""
    console.print(
        Panel.fit(
            "[bold blue]SmolVLM Image Description[/bold blue]\n"
            f"Using model: [green]{model}[/green]",
            border_style="blue",
        )
    )

    # Function to process images using SmolVLM
    async def process_images():
        # Create the image description service
        image_service = SmolVLMImageDescriptionService(
            model_name=model, min_image_size=min_size
        )

        # Find images to process
        if doc_path:
            # Extract images from document
            with open(doc_path, "r") as f:
                document_content = f.read()

            # Fix path issues in document content
            adjusted_content = document_content.replace("![](output/images/", "![](")

            console.print(f"Extracting images from [bold]{doc_path}[/bold]...")
            image_data = await image_service.extract_image_contexts(
                adjusted_content, image_dir
            )

            # Limit the number of images processed if needed
            if sample_limit and len(image_data) > sample_limit:
                console.print(
                    f"Limiting to {sample_limit} images from {len(image_data)} found"
                )
                image_data = image_data[:sample_limit]

            if not image_data:
                console.print("[bold red]No images found in document[/bold red]")
                return {}

            console.print(
                f"Found [bold green]{len(image_data)}[/bold green] images in document"
            )

        else:
            # Scan directory for images
            console.print(f"Scanning directory: [bold]{image_dir}[/bold]")

            image_files = []
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith((".png", ".gif", ".jpg", ".jpeg")):
                        image_files.append(os.path.join(root, file))

            # Limit the number of images processed if needed
            if sample_limit and len(image_files) > sample_limit:
                console.print(
                    f"Limiting to {sample_limit} images from {len(image_files)} found"
                )
                image_files = image_files[:sample_limit]

            if not image_files:
                console.print("[bold red]No images found in directory[/bold red]")
                return {}

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
                "Generating SmolVLM image descriptions...", total=len(image_data)
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

        return results

    # Run the async function
    results = asyncio.run(process_images())

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


@image_app.command("compare")
def compare_descriptions(
    image_dir: str = typer.Option(
        settings.images_dir,
        "--image-dir",
        help="Directory containing images to process",
    ),
    min_size: int = typer.Option(
        64, "--min-size", help="Minimum size in pixels for images to process"
    ),
    gemma_model: str = typer.Option(
        "gemma3:27b", "--gemma-model", help="Ollama model to use for image description"
    ),
    smolvlm_model: str = typer.Option(
        "HuggingFaceTB/SmolVLM-Synthetic",
        "--smolvlm-model",
        help="HuggingFace model to use for image description",
    ),
    sample_limit: int = typer.Option(
        3, "--sample-limit", help="Maximum number of images to process"
    ),
    doc_path: str = typer.Option(
        "test/samples/renew-a-certificate.md",
        "--doc-path",
        help="Path to a markdown document to extract images from",
    ),
):
    """Compare image descriptions between Gemma and SmolVLM models."""
    console.print(
        Panel.fit(
            "[bold blue]Image Description Comparison[/bold blue]\n"
            f"Gemma model: [green]{gemma_model}[/green]\n"
            f"SmolVLM model: [green]{smolvlm_model}[/green]",
            border_style="blue",
        )
    )

    # Function to process images using both models
    async def process_images():
        # Create the image description services
        gemma_service = OllamaImageDescriptionService(
            settings=settings.llm, model=gemma_model, min_image_size=min_size
        )

        smolvlm_service = SmolVLMImageDescriptionService(
            model_name=smolvlm_model, min_image_size=min_size
        )

        # Extract images from document
        with open(doc_path, "r") as f:
            document_content = f.read()

        # Fix path issues in document content
        adjusted_content = document_content.replace("![](output/images/", "![](")

        console.print(f"Extracting images from [bold]{doc_path}[/bold]...")

        # We'll use Gemma service for extraction since both use the same method
        image_data = await gemma_service.extract_image_contexts(
            adjusted_content, image_dir
        )

        # Limit the number of images processed if needed
        if sample_limit and len(image_data) > sample_limit:
            console.print(
                f"Limiting to {sample_limit} images from {len(image_data)} found"
            )
            image_data = image_data[:sample_limit]

        if not image_data:
            console.print("[bold red]No images found in document[/bold red]")
            return {}

        console.print(
            f"Found [bold green]{len(image_data)}[/bold green] images in document"
        )

        # Process images with both models
        results = {}

        for i, (img_path, context) in enumerate(image_data):
            basename = os.path.basename(img_path)
            console.print(
                f"\n[bold]Processing image {i+1}/{len(image_data)}:[/bold] {basename}"
            )

            # Generate Gemma description
            console.print("  Generating Gemma description...")
            gemma_desc = await gemma_service.generate_image_description(
                img_path, context
            )

            # Generate SmolVLM description
            console.print("  Generating SmolVLM description...")
            smolvlm_desc = await smolvlm_service.generate_image_description(
                img_path, context
            )

            results[basename] = {
                "path": img_path,
                "gemma": gemma_desc or "No description generated",
                "smolvlm": smolvlm_desc or "No description generated",
                "context": context,
            }

        return results

    # Run the async function
    results = asyncio.run(process_images())

    # Display results
    console.print(
        f"\n[bold green]Generated descriptions for {len(results)} images[/bold green]"
    )

    for img_name, data in results.items():
        console.print(
            Panel(f"[bold cyan]Image:[/bold cyan] {img_name}", border_style="cyan")
        )

        if data["context"]:
            console.print("[bold]Image Context:[/bold]")
            console.print(Markdown(data["context"]))
            console.print()

        table = Table(title="Description Comparison")
        table.add_column("Model", style="bold blue")
        table.add_column("Description")

        table.add_row("Gemma", data["gemma"])
        table.add_row("SmolVLM", data["smolvlm"])

        console.print(table)
        console.print()


def register_commands(app: typer.Typer):
    """Register image commands with the main app."""
    app.add_typer(
        image_app, name="images", help="Image description and processing commands"
    )
