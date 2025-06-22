#!/usr/bin/env python3
"""
Strip images and links from markdown files.

This script removes:
- Images: ![alt text](url)
- Links: [text](url)
"""

import re
import argparse
from pathlib import Path


def strip_markdown_images(content: str) -> str:
    """Remove all markdown images from content."""
    # Pattern matches ![alt text](url) including optional title
    image_pattern = r"!\[([^\]]*)\]\([^)]+\)"
    return re.sub(image_pattern, "", content)


def strip_markdown_links(content: str, keep_text: bool = True) -> str:
    """Remove markdown links, optionally keeping the link text."""
    # Pattern matches [text](url)
    link_pattern = r"\[([^\]]+)\]\([^)]+\)"

    if keep_text:
        # Replace with just the link text
        return re.sub(link_pattern, r"\1", content)
    else:
        # Remove entirely
        return re.sub(link_pattern, "", content)


def strip_reference_style_links(content: str) -> str:
    """Remove reference-style link definitions."""
    # Pattern matches [id]: url "optional title"
    ref_pattern = r"^\s*\[[^\]]+\]:\s*.+$"
    return re.sub(ref_pattern, "", content, flags=re.MULTILINE)


def clean_empty_lines(content: str) -> str:
    """Clean up multiple consecutive empty lines."""
    # Replace 3 or more newlines with 2
    content = re.sub(r"\n{3,}", "\n\n", content)
    # Remove trailing whitespace on each line
    content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)
    return content.strip()


def process_file(
    input_path: Path,
    output_path: Path,
    strip_images: bool = True,
    strip_links: bool = True,
    keep_link_text: bool = True,
) -> dict:
    """Process a single markdown file."""

    # Read input file
    content = input_path.read_text(encoding="utf-8")
    original_size = len(content)

    # Strip images if requested
    if strip_images:
        content = strip_markdown_images(content)

    # Strip links if requested
    if strip_links:
        content = strip_markdown_links(content, keep_text=keep_link_text)
        content = strip_reference_style_links(content)

    # Clean up empty lines
    content = clean_empty_lines(content)

    # Write output
    output_path.write_text(content, encoding="utf-8")

    final_size = len(content)

    return {
        "original_size": original_size,
        "final_size": final_size,
        "reduction": original_size - final_size,
        "reduction_percent": (
            ((original_size - final_size) / original_size * 100)
            if original_size > 0
            else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Strip images and links from markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Strip both images and links (keep link text)
  python strip_markdown_media.py input.md output.md
  
  # Strip only images
  python strip_markdown_media.py input.md output.md --no-strip-links
  
  # Strip links but remove text too
  python strip_markdown_media.py input.md output.md --no-keep-link-text
  
  # Process all markdown files in a directory
  python strip_markdown_media.py input_dir/ output_dir/ --recursive
""",
    )

    parser.add_argument("input", type=Path, help="Input markdown file or directory")
    parser.add_argument("output", type=Path, help="Output markdown file or directory")
    parser.add_argument(
        "--no-strip-images", action="store_true", help="Don't strip images"
    )
    parser.add_argument(
        "--no-strip-links", action="store_true", help="Don't strip links"
    )
    parser.add_argument(
        "--no-keep-link-text",
        action="store_true",
        help="Remove link text along with URLs (default: keep text)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Process directories recursively"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Determine what to strip
    strip_images = not args.no_strip_images
    strip_links = not args.no_strip_links
    keep_link_text = not args.no_keep_link_text

    if not strip_images and not strip_links:
        print("Nothing to strip! Use --no-strip-images OR --no-strip-links, not both.")
        return

    # Handle single file
    if args.input.is_file():
        if args.output.is_dir():
            output_path = args.output / args.input.name
        else:
            output_path = args.output

        if args.dry_run:
            print(f"Would process: {args.input} -> {output_path}")
            return

        stats = process_file(
            args.input, output_path, strip_images, strip_links, keep_link_text
        )

        print(f"Processed: {args.input}")
        print(f"  Original size: {stats['original_size']:,} bytes")
        print(f"  Final size: {stats['final_size']:,} bytes")
        print(
            f"  Reduction: {stats['reduction']:,} bytes ({stats['reduction_percent']:.1f}%)"
        )
        print(f"  Output: {output_path}")

    # Handle directory
    elif args.input.is_dir():
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=True)

        # Find all markdown files
        if args.recursive:
            md_files = list(args.input.rglob("*.md"))
        else:
            md_files = list(args.input.glob("*.md"))

        if not md_files:
            print(f"No markdown files found in {args.input}")
            return

        print(f"Found {len(md_files)} markdown files")

        total_original = 0
        total_final = 0

        for md_file in md_files:
            # Calculate relative path for output
            rel_path = md_file.relative_to(args.input)
            output_path = args.output / rel_path

            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                print(f"Would process: {md_file} -> {output_path}")
                continue

            stats = process_file(
                md_file, output_path, strip_images, strip_links, keep_link_text
            )
            total_original += stats["original_size"]
            total_final += stats["final_size"]

            print(
                f"Processed: {rel_path} ({stats['reduction_percent']:.1f}% reduction)"
            )

        if not args.dry_run:
            print(f"\nTotal statistics:")
            print(f"  Original size: {total_original:,} bytes")
            print(f"  Final size: {total_final:,} bytes")
            print(
                f"  Total reduction: {total_original - total_final:,} bytes ({(total_original - total_final) / total_original * 100:.1f}%)"
            )

    else:
        print(f"Error: {args.input} not found")
        return


if __name__ == "__main__":
    main()

