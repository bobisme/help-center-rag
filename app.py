#!/usr/bin/env python3

"""CLI application for converting Epic documentation HTML to Markdown."""

import argparse
import os
import sys
import json
from typing import Dict, Any, Optional

from html2md import convert_html_to_markdown, preprocess_html
from html2md.loaders import load_from_html_file, load_from_json_file, get_page_count


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HTML to Markdown from epic-docs.json"
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Index of the page to convert (default: 0)"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="output/images",
        help="Directory containing images (default: output/images). "
        "Set to empty string to disable image processing.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/epic-docs.json",
        help="Path to the input JSON file (default: output/epic-docs.json)",
    )
    parser.add_argument(
        "--html-file",
        type=str,
        help="Path to an HTML file to process directly instead of using JSON",
    )
    parser.add_argument(
        "--save-html",
        type=str,
        help="Save the preprocessed HTML to this file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the markdown output (default: print to stdout)",
    )
    parser.add_argument(
        "--list-pages",
        action="store_true",
        help="List all available pages in the JSON file with their indices",
    )
    return parser.parse_args()


def main() -> None:
    """Run the HTML to Markdown conversion CLI application."""
    args = parse_args()

    # Check images directory
    images_dir = args.images if args.images else None
    if images_dir and not os.path.exists(images_dir):
        sys.stderr.write(
            f"Warning: Images directory '{images_dir}' not found. "
            "Image links may be broken.\n"
        )

    try:
        # List pages if requested
        if args.list_pages and not args.html_file:
            list_pages(args.input)
            return

        # Get HTML content
        if args.html_file:
            raw_html = load_from_html_file(args.html_file)
            title = None
        else:
            raw_html, title, _ = load_from_json_file(args.input, args.index)
            if title:
                sys.stderr.write(f"Processing page: {title}\n")

        # Process HTML and convert to markdown
        processed_html = preprocess_html(raw_html, images_dir)

        # Save preprocessed HTML if requested
        if args.save_html:
            with open(args.save_html, "w") as f:
                f.write(processed_html)
                sys.stderr.write(f"Preprocessed HTML saved to {args.save_html}\n")

        # Convert to markdown
        markdown = convert_html_to_markdown(
            raw_html, images_dir=images_dir, heading_style="ATX", wrap=True
        )

        # Output markdown
        if args.output:
            with open(args.output, "w") as f:
                f.write(markdown)
                sys.stderr.write(f"Markdown saved to {args.output}\n")
        else:
            print(markdown)

    except FileNotFoundError:
        file_path = args.html_file if args.html_file else args.input
        sys.stderr.write(f"Error: File {file_path} not found\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Invalid JSON format in {args.input}\n")
        sys.exit(1)
    except ValueError as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)


def list_pages(json_path: str) -> None:
    """List all pages in the JSON file with their indices."""
    try:
        with open(json_path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            sys.stderr.write(f"Error: Invalid JSON structure in {json_path}\n")
            sys.exit(1)

        print(f"Available pages in {json_path}:")
        print(f"Total pages: {len(data['pages'])}")
        print("-" * 80)
        print(f"{'INDEX':<8} {'TITLE'}")
        print("-" * 80)
        
        for i, page in enumerate(data["pages"]):
            title = page.get("title", "Untitled")
            print(f"{i:<8} {title}")

    except FileNotFoundError:
        sys.stderr.write(f"Error: File {json_path} not found\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Invalid JSON format in {json_path}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()