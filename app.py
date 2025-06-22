#!/usr/bin/env python3

"""CLI application for converting Epic documentation HTML to Markdown."""

import os
import sys
import json
from typing import Dict, Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from html2md import convert_html_to_markdown, preprocess_html
from html2md.loaders import load_from_html_file, load_from_json_file


# Create Typer app
app = typer.Typer(
    help="Convert Epic documentation HTML to Markdown.",
    add_completion=False,
    # Disable pretty exceptions to avoid stdout pollution
    pretty_exceptions_enable=False,
)

# Create console for rich output - send to stderr so it doesn't interfere with
# markdown output
console = Console(stderr=True)


def is_error_page(markdown: str, title: str) -> bool:
    """Detect if a page is an error page and should be skipped."""
    error_indicators = [
        "AccessDenied",
        "Access Denied",
        "<Error>",
        "<Code>AccessDenied</Code>",
        "This XML file does not appear to have any style information",
        "The document tree is shown below",
        "RequestId",
        "HostId",
    ]

    # Check for untitled pages with error content
    if title.lower() in ["untitled", "untitled page"] and any(
        indicator in markdown for indicator in error_indicators
    ):
        return True

    # Check for pages that are mostly error content
    error_count = sum(1 for indicator in error_indicators if indicator in markdown)
    if error_count >= 3:  # Multiple error indicators suggest it's an error page
        return True

    # Check if the page is very short and contains error indicators
    if len(markdown.strip()) < 500 and any(
        indicator in markdown for indicator in error_indicators
    ):
        return True

    return False


@app.command("convert")
def convert(
    index: int = typer.Option(0, "--index", "-i", help="Index of the page to convert"),
    all_pages: bool = typer.Option(
        False, "--all", help="Convert all pages in the JSON file"
    ),
    images_dir: str = typer.Option(
        "output/images",
        "--images",
        "-img",
        help='Directory containing images. Set to "" to disable image processing.',
    ),
    input_file: str = typer.Option(
        "output/epic-docs.json", "--input", "-in", help="Path to the input JSON file"
    ),
    html_file: Optional[str] = typer.Option(
        None,
        "--html-file",
        "-html",
        help="Path to an HTML file to process directly instead of using JSON",
    ),
    save_html: Optional[str] = typer.Option(
        None, "--save-html", "-sh", help="Save the preprocessed HTML to this file path"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the markdown output (default: print to stdout)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-od",
        help="Directory to save individual markdown files when using --all",
    ),
    heading_style: str = typer.Option(
        "ATX", "--heading-style", "-hs", help="Heading style: ATX (#) or SETEXT (===)"
    ),
    wrap: bool = typer.Option(
        True, "--wrap/--no-wrap", help="Whether to wrap long lines"
    ),
):
    """Convert HTML to Markdown."""
    # Validate options
    if all_pages and html_file:
        console.print(
            "[red]Error:[/red] Cannot use --all with --html-file", style="bold red"
        )
        raise typer.Exit(code=1)

    # When using --all with --output-dir, require output_dir to be specified
    if (
        all_pages
        and output_dir
        and not os.path.exists(
            os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
        )
    ):
        console.print(
            f"[red]Error:[/red] Parent directory for --output-dir does not exist",
            style="bold red",
        )
        raise typer.Exit(code=1)

    if all_pages and output_dir and output_file:
        console.print(
            "[red]Error:[/red] Cannot use both --output-dir and --output with --all",
            style="bold red",
        )
        raise typer.Exit(code=1)

    # Check images directory
    if images_dir and not os.path.exists(images_dir):
        console.print(
            f"[yellow]Warning:[/yellow] Images directory '{images_dir}' not found. "
            "Image links may be broken."
        )

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        console.print(f"Created output directory: [cyan]{output_dir}[/cyan]")

    try:
        if all_pages:
            # Process all pages
            with open(input_file, "r") as f:
                data: Dict[str, Any] = json.load(f)

            if not data.get("pages") or not isinstance(data["pages"], list):
                console.print(
                    f"[red]Error:[/red] Invalid JSON structure in {input_file}"
                )
                raise typer.Exit(code=1)

            pages = data["pages"]
            total_pages = len(pages)

            console.print(
                f"Converting [green]{total_pages}[/green] pages from [cyan]{input_file}[/cyan]"
            )

            if output_dir:
                # Individual files output
                with Progress(console=console) as progress:
                    task = progress.add_task(
                        "[green]Converting pages...", total=total_pages
                    )

                    for i, page in enumerate(pages):
                        try:
                            # Load content for this page
                            raw_html, title, _ = load_from_json_file(input_file, i)

                            # Convert to markdown
                            markdown = convert_html_to_markdown(
                                raw_html,
                                images_dir=images_dir,
                                heading_style=heading_style,
                                wrap=wrap,
                            )

                            # Skip error pages
                            if is_error_page(markdown, title):
                                console.print(
                                    f"[yellow]Skipping error page:[/yellow] {title}"
                                )
                                progress.update(task, advance=1)
                                continue

                            # Generate safe filename
                            safe_title = "".join(
                                c for c in title if c.isalnum() or c in (" ", "-", "_")
                            ).rstrip()
                            safe_title = safe_title.replace(" ", "_")[
                                :50
                            ]  # Limit length
                            filename = f"{i:03d}_{safe_title}.md"
                            filepath = os.path.join(output_dir, filename)

                            # Save markdown
                            with open(filepath, "w") as f:
                                f.write(markdown)

                            progress.update(
                                task,
                                advance=1,
                                description=f"[green]Converted: {title[:30]}...",
                            )

                        except Exception as e:
                            console.print(
                                f"[yellow]Warning:[/yellow] Failed to convert page {i}: {e}"
                            )
                            progress.update(task, advance=1)
                            continue

                console.print(
                    f"[green]Conversion complete![/green] Files saved to [cyan]{output_dir}[/cyan]"
                )

            else:
                # Single file output (to file or stdout)
                all_markdown = []

                with Progress(console=console) as progress:
                    task = progress.add_task(
                        "[green]Converting pages...", total=total_pages
                    )

                    for i, page in enumerate(pages):
                        try:
                            # Load content for this page
                            raw_html, title, _ = load_from_json_file(input_file, i)

                            # Convert to markdown
                            markdown = convert_html_to_markdown(
                                raw_html,
                                images_dir=images_dir,
                                heading_style=heading_style,
                                wrap=wrap,
                            )

                            # Skip error pages
                            if is_error_page(markdown, title):
                                console.print(
                                    f"[yellow]Skipping error page:[/yellow] {title}"
                                )
                                progress.update(task, advance=1)
                                continue

                            # Add separator between pages (don't add title since it's already in the markdown)
                            page_markdown = f"{markdown}\n\n---\n\n"
                            all_markdown.append(page_markdown)

                            progress.update(
                                task,
                                advance=1,
                                description=f"[green]Converted: {title[:30]}...",
                            )

                        except Exception as e:
                            console.print(
                                f"[yellow]Warning:[/yellow] Failed to convert page {i}: {e}"
                            )
                            progress.update(task, advance=1)
                            continue

                # Write combined markdown
                combined_markdown = "".join(all_markdown)

                if output_file:
                    with open(output_file, "w") as f:
                        f.write(combined_markdown)
                    console.print(
                        f"[green]Conversion complete![/green] Combined file saved to [cyan]{output_file}[/cyan]"
                    )
                else:
                    # Print to stdout
                    sys.stdout.write(combined_markdown)
                    console.print(
                        f"[green]Conversion complete![/green] Combined markdown printed to stdout"
                    )

        else:
            # Process single page (existing logic)
            with Progress(console=console) as progress:
                task = progress.add_task("[green]Processing...", total=3)

                # Step 1: Load content
                if html_file:
                    raw_html = load_from_html_file(html_file)
                    title = os.path.basename(html_file)
                    console.print(f"Loading HTML from [cyan]{html_file}[/cyan]")
                else:
                    raw_html, title, _ = load_from_json_file(input_file, index)
                    console.print(
                        f"Processing page: [cyan]{title}[/cyan] (index: {index})"
                    )

                progress.update(
                    task, advance=1, description="[green]Preprocessing HTML..."
                )

                # Step 2: Preprocess HTML
                processed_html = preprocess_html(raw_html, images_dir)

                # Save preprocessed HTML if requested
                if save_html:
                    with open(save_html, "w") as f:
                        f.write(processed_html)
                    console.print(
                        f"Preprocessed HTML saved to [cyan]{save_html}[/cyan]"
                    )

                progress.update(
                    task, advance=1, description="[green]Converting to Markdown..."
                )

                # Step 3: Convert to markdown
                markdown = convert_html_to_markdown(
                    raw_html,
                    images_dir=images_dir,
                    heading_style=heading_style,
                    wrap=wrap,
                )

                progress.update(
                    task, advance=1, description="[green]Conversion complete!"
                )

            # Output markdown
            if output_file:
                with open(output_file, "w") as f:
                    f.write(markdown)
                console.print(
                    f"Markdown saved to [cyan]{output_file}[/cyan]", style="green"
                )
            else:
                # Print markdown directly to stdout, not through the console
                # This ensures clean output for piping to files
                sys.stdout.write(markdown)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] File not found - {e}", style="bold red")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        console.print(
            f"[red]Error:[/red] Invalid JSON format in {input_file}", style="bold red"
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {str(e)}", style="bold red")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}", style="bold red")
        raise typer.Exit(code=1)


@app.command("list")
def list_pages(
    input_file: str = typer.Option(
        "output/epic-docs.json", "--input", "-in", help="Path to the input JSON file"
    ),
    limit: int = typer.Option(
        0, "--limit", "-l", help="Limit the number of pages shown (0 = show all)"
    ),
):
    """List all pages in the JSON file with their indices."""
    try:
        with open(input_file, "r") as f:
            data: Dict[str, Any] = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            console.print(f"[red]Error:[/red] Invalid JSON structure in {input_file}")
            raise typer.Exit(code=1)

        pages = data["pages"]
        page_count = len(pages)

        console.print(f"Available pages in [cyan]{input_file}[/cyan]:")
        console.print(f"Total pages: [green]{page_count}[/green]")

        # Create a table
        table = Table(show_header=True, header_style="bold")
        table.add_column("INDEX", style="dim", width=8)
        table.add_column("TITLE", style="cyan")

        # Decide how many pages to show
        if limit <= 0 or limit > page_count:
            display_pages = pages
        else:
            display_pages = pages[:limit]

        # Add rows to the table
        for i, page in enumerate(display_pages):
            title = page.get("title", "Untitled")
            table.add_row(str(i), title)

        console.print(table)

        if limit > 0 and limit < page_count:
            console.print(
                f"Showing {limit} of {page_count} pages. Use --limit 0 to see all."
            )

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File {input_file} not found")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        console.print(f"[red]Error:[/red] Invalid JSON format in {input_file}")
        raise typer.Exit(code=1)


@app.command("info")
def show_info():
    """Show information about the tool."""
    console.print("[bold cyan]Epic HTML to Markdown Converter[/bold cyan]")
    console.print("A tool for converting Epic documentation from HTML to Markdown.")
    console.print("\n[bold]Features:[/bold]")

    features = [
        "Convert HTML to clean, well-formatted Markdown",
        "Process nested lists correctly",
        "Convert inline styling to proper Markdown formatting",
        "Handle images with local references",
        "Clean up navigation and context elements",
    ]

    for feature in features:
        console.print(f"• [green]{feature}[/green]")

    console.print("\n[bold]Commands:[/bold]")
    console.print("• [yellow]convert[/yellow]: Convert HTML to Markdown")
    console.print("• [yellow]list[/yellow]: List available pages in the JSON file")
    console.print("• [yellow]info[/yellow]: Show information about the tool")

    console.print(
        "\nFor more details on each command, use: [cyan]app.py COMMAND --help[/cyan]"
    )


if __name__ == "__main__":
    app()
