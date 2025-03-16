#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md


def preprocess_html(html: str, images_dir: Optional[str]) -> str:
    """Remove header elements and other unnecessary content from the HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove logo image
    for img in soup.find_all("img", src=lambda src: bool(src) and "logo.png" in src):
        img.decompose()

    # Remove "Applied Epic July 2023 Help File" text
    for text in soup.find_all(string="Applied Epic July 2023 Help File"):
        if text.parent:
            text.parent.decompose()

    # Remove "Click here to see this page in full context" text
    for element in soup.find_all(
        string=lambda text: bool(text)
        and "Click here to see this page in full context" in text
    ):
        if element.parent:
            element.parent.decompose()

    # Replace links with their text content
    for link in soup.find_all("a"):
        # Make sure we're dealing with a Tag, not NavigableString
        if not isinstance(link, Tag):
            continue

        # If the link is javascript or a local file reference, replace it with
        # just the text
        href = link.get("href", "")
        if isinstance(href, str) and (
            href.startswith("javascript:")
            or href.endswith(".htm")
            or href.startswith("../")
        ):
            # Preserve the inner text
            link_text = link.get_text(strip=True)
            # Create a new string element to replace the link
            new_element = soup.new_string(link_text)
            link.replace_with(new_element)

    # Handle images
    for img in soup.find_all("img"):
        # Make sure we're dealing with a Tag, not NavigableString
        if not isinstance(img, Tag):
            continue

        src = img.get("src", "")
        if isinstance(src, str) and src and not src.startswith("data:"):
            # Skip the logo.png which we removed above
            if "logo.png" in src:
                continue

            # If no images directory specified, remove all images
            if not images_dir:
                img.decompose()
                continue

            # Extract the filename from the URL
            filename = os.path.basename(src)

            # Look for the file in the images directory
            if os.path.exists(images_dir):
                # The files in images_dir have base64-encoded prefixes
                # Find a matching file based on the end of the filename
                for img_file in os.listdir(images_dir):
                    if img_file.endswith(filename) or filename in img_file:
                        # Update the src to point to the local file
                        img["src"] = f"{images_dir}/{img_file}"
                        break
                else:
                    # If not found directly, look for a file that might match
                    # based on content
                    found = False
                    for img_file in os.listdir(images_dir):
                        if any(
                            part in img_file.lower()
                            for part in filename.lower().split(".")
                        ):
                            img["src"] = f"{images_dir}/{img_file}"
                            found = True
                            break

                    # If still not found, consider removing the image
                    if (
                        not found
                        and isinstance(src, str)
                        and not src.startswith("http")
                    ):
                        # For non-http sources that we can't resolve, we'll
                        # remove the image
                        img.decompose()

    return str(soup)


def main() -> None:
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
        help=(
            "Directory containing images (default: output/images). "
            "Set to empty string to disable image processing."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/epic-docs.json",
        help="Path to the input JSON file (default: output/epic-docs.json)",
    )
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            data: Dict[str, Any] = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            sys.stderr.write("Error: Invalid JSON structure - 'pages' list not found\n")
            sys.exit(1)

        if args.index < 0 or args.index >= len(data["pages"]):
            sys.stderr.write(
                f"Error: Index {args.index} out of range (0-{len(data['pages'])-1})\n"
            )
            sys.exit(1)

        page = data["pages"][args.index]
        raw_html = page.get("rawHtml", "")

        if not raw_html:
            sys.stderr.write(f"Error: No HTML content found at index {args.index}\n")
            sys.exit(1)

        # Preprocess the HTML to remove unwanted elements
        images_dir = args.images if args.images else None

        # Check if the images directory exists
        if images_dir and not os.path.exists(images_dir):
            sys.stderr.write(
                f"Warning: Images directory '{images_dir}' not found. "
                "Image links may be broken.\n"
            )

        processed_html = preprocess_html(raw_html, images_dir)

        # Convert to markdown
        markdown = md(processed_html, heading_style="ATX", wrap=True)
        print(markdown)

    except FileNotFoundError:
        sys.stderr.write(f"Error: File {args.input} not found\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Invalid JSON format in {args.input}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
