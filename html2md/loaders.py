"""Loaders for Epic documentation data."""

import json
import sys
from typing import Dict, Any, List, Tuple, Optional


def load_from_html_file(file_path: str) -> str:
    """
    Load HTML content from a file.

    Args:
        file_path: Path to the HTML file

    Returns:
        The HTML content as a string
    """
    with open(file_path, "r") as f:
        return f.read()


def load_from_json_file(
    json_path: str, index: int
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """
    Load HTML content from a JSON file containing Epic documentation.

    Args:
        json_path: Path to the JSON file
        index: Index of the page to retrieve

    Returns:
        Tuple of (HTML content, page title, list of all pages)

    Raises:
        ValueError: If the JSON structure is invalid or index is out of range
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(json_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    if not data.get("pages") or not isinstance(data["pages"], list):
        raise ValueError("Invalid JSON structure - 'pages' list not found")

    pages = data["pages"]
    
    if index < 0 or index >= len(pages):
        raise ValueError(f"Index {index} out of range (0-{len(pages)-1})")

    page = pages[index]
    raw_html = page.get("rawHtml", "")
    title = page.get("title", None)

    if not raw_html:
        raise ValueError(f"No HTML content found at index {index}")

    return raw_html, title, pages


def get_page_count(json_path: str) -> int:
    """
    Get the number of pages in the JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Number of pages
    """
    try:
        with open(json_path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        if not data.get("pages") or not isinstance(data["pages"], list):
            return 0

        return len(data["pages"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0