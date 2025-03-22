#!/usr/bin/env python3
"""Integration tests for the html2md module."""

import os
import json
import tempfile
import pytest
from html2md import convert_html_to_markdown
from html2md.loaders import load_from_json_file, get_page_count


@pytest.fixture
def sample_json_file():
    """Create a sample JSON file with Epic doc format for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_data = {
            "metadata": {
                "crawlDate": "2023-01-01T00:00:00Z",
                "baseUrl": "https://test.com",
                "totalPages": 1,
                "maxDepth": 3
            },
            "pages": [
                {
                    "title": "Test Page with Images",
                    "url": "https://test.com/test",
                    "rawHtml": """
                    <html>
                        <body>
                            <h1>Test Document</h1>
                            <p>This is a test paragraph with an image: <img src="test.jpg" alt="Test Image"></p>
                            <p>And here's another image: <img src="logo.png" alt="Logo"></p>
                            <div class="Banner">This should be removed</div>
                        </body>
                    </html>
                    """,
                    "metadata": {
                        "depth": 1,
                        "path": ["Test"],
                        "crawlDate": "2023-01-01T00:00:00Z"
                    },
                    "images": [
                        {
                            "originalUrl": "https://test.com/test.jpg",
                            "localPath": "output/images/test.jpg",
                            "alt": "Test Image",
                            "width": 500,
                            "height": 400,
                            "isScreenshot": True
                        }
                    ]
                }
            ]
        }
        json.dump(json_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def images_dir():
    """Create a temporary directory for test images."""
    test_dir = "output/images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test image file
    with open(os.path.join(test_dir, "test.jpg"), "w") as f:
        f.write("dummy image content")
        
    yield test_dir


def test_end_to_end_conversion(sample_json_file, images_dir):
    """Test the end-to-end process of loading JSON and converting to markdown."""
    # Get the HTML content from the JSON file
    html_content, title, _ = load_from_json_file(sample_json_file, 0)
    
    # Convert the HTML to markdown
    markdown = convert_html_to_markdown(html_content, images_dir=images_dir)
    
    # Check that the markdown conversion worked correctly
    assert "# Test Document" in markdown
    assert "This is a test paragraph with an image" in markdown
    assert "![Test Image]" in markdown
    assert "output/images" in markdown
    
    # Check that logo image is removed (logo.png is handled by _remove_header_elements)
    assert "![Logo]" not in markdown  # Logo image should be removed
    
    # Check title
    assert title == "Test Page with Images"


def test_end_to_end_without_images_dir(sample_json_file):
    """Test end-to-end conversion with a non-existent images directory."""
    # Get the HTML content from the JSON file
    html_content, _, _ = load_from_json_file(sample_json_file, 0)
    
    # Convert with non-existent directory
    markdown = convert_html_to_markdown(html_content, images_dir="nonexistent_dir")
    
    # Check that images were removed
    assert "![Test Image]" not in markdown
    assert "![Logo]" not in markdown
    
    # But text content should still be there
    assert "# Test Document" in markdown
    assert "This is a test paragraph with an image" in markdown