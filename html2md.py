#!/usr/bin/env python3

import argparse
import json
import sys
from markdownify import markdownify as md


def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML to Markdown from epic-docs.json"
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Index of the page to convert (default: 0)"
    )
    args = parser.parse_args()

    try:
        with open("output/epic-docs.json", "r") as f:
            data = json.load(f)

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

        markdown = md(raw_html, heading_style="ATX", wrap=True)
        print(markdown)

    except FileNotFoundError:
        sys.stderr.write("Error: File output/epic-docs.json not found\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write("Error: Invalid JSON format in output/epic-docs.json\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

