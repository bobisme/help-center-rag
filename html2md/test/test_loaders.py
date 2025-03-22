#!/usr/bin/env python3
"""Unit tests for HTML to Markdown loaders."""

import os
import json
import tempfile
import pytest
from html2md.loaders import load_from_html_file, load_from_json_file, get_page_count


@pytest.fixture
def sample_html_file():
    """Create a sample HTML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write("<html><body><h1>Test HTML</h1><p>Test paragraph</p></body></html>")
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_json_file():
    """Create a sample JSON file with Epic doc format for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_data = {
            "metadata": {
                "crawlDate": "2023-01-01T00:00:00Z",
                "baseUrl": "https://test.com",
                "totalPages": 2,
                "maxDepth": 3
            },
            "pages": [
                {
                    "title": "Page 1",
                    "url": "https://test.com/page1",
                    "rawHtml": "<h1>Page 1</h1><p>Content 1</p>",
                    "metadata": {
                        "depth": 1,
                        "path": ["Test", "Page 1"],
                        "crawlDate": "2023-01-01T00:00:00Z"
                    }
                },
                {
                    "title": "Page 2",
                    "url": "https://test.com/page2",
                    "rawHtml": "<h1>Page 2</h1><p>Content 2</p>",
                    "metadata": {
                        "depth": 1,
                        "path": ["Test", "Page 2"],
                        "crawlDate": "2023-01-01T00:00:00Z"
                    }
                }
            ]
        }
        json.dump(json_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_load_from_html_file(sample_html_file):
    """Test loading HTML content from a file."""
    content = load_from_html_file(sample_html_file)
    assert "<h1>Test HTML</h1>" in content
    assert "<p>Test paragraph</p>" in content


def test_load_from_json_file(sample_json_file):
    """Test loading HTML content from a JSON file with Epic doc format."""
    html, title, pages = load_from_json_file(sample_json_file, 0)
    
    # Check HTML content
    assert "<h1>Page 1</h1>" in html
    assert "<p>Content 1</p>" in html
    
    # Check title
    assert title == "Page 1"
    
    # Check pages list
    assert len(pages) == 2
    assert pages[1]["title"] == "Page 2"


def test_load_from_json_file_with_index(sample_json_file):
    """Test loading HTML content from a specific index in a JSON file."""
    html, title, _ = load_from_json_file(sample_json_file, 1)
    
    # Check HTML content and title for second page
    assert "<h1>Page 2</h1>" in html
    assert "<p>Content 2</p>" in html
    assert title == "Page 2"


def test_load_from_json_file_invalid_index(sample_json_file):
    """Test loading with an invalid index raises ValueError."""
    with pytest.raises(ValueError, match=r"Index .* out of range"):
        load_from_json_file(sample_json_file, 10)


def test_get_page_count(sample_json_file):
    """Test getting the page count from a JSON file."""
    page_count = get_page_count(sample_json_file)
    assert page_count == 2


def test_get_page_count_nonexistent_file():
    """Test getting page count from a nonexistent file returns 0."""
    page_count = get_page_count("nonexistent_file.json")
    assert page_count == 0