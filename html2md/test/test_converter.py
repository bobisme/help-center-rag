#!/usr/bin/env python3
"""Unit tests for HTML to Markdown conversion with image handling using pytest."""

import os
import shutil
import pytest
from html2md import convert_html_to_markdown, preprocess_html


@pytest.fixture
def test_images_dir():
    """Set up a temporary directory with test images."""
    dir_path = "test_images_dir"
    os.makedirs(dir_path, exist_ok=True)

    # Create a test image file
    with open(os.path.join(dir_path, "test_image.jpg"), "w") as f:
        f.write("dummy image content")

    yield dir_path

    # Clean up
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def test_default_image_directory():
    """Test conversion with default image directory (output/images)."""
    html = '<p>Test with <img src="test.jpg" alt="Test Image"></p>'
    result = convert_html_to_markdown(html)
    assert "output/images" in result
    # The image will be kept in the markdown, with potentially a different filename
    assert "![Test Image]" in result


def test_nonexistent_image_directory():
    """Test conversion with non-existent image directory."""
    html = '<p>Test with <img src="test.jpg" alt="Test Image"></p>'
    result = convert_html_to_markdown(html, images_dir="nonexistent_dir")
    # Image should be removed (not found in output)
    assert "![Test Image]" not in result
    # The exact string may vary based on markdownify's behavior, just check image is removed
    assert result.strip() == "Test with"


def test_custom_image_directory(test_images_dir):
    """Test conversion with custom image directory."""
    html = '<p>Test with <img src="test_image.jpg" alt="Test Image"></p>'
    result = convert_html_to_markdown(html, images_dir=test_images_dir)
    # Image should be kept with reference to the custom directory
    assert f"{test_images_dir}/" in result
    assert "![Test Image]" in result


def test_image_size_filtering():
    """Test that images are processed correctly based on size."""
    # This simulates HTML with image dimensions
    html = """
    <p>Large image: <img src="large.jpg" alt="Large" width="800" height="600"></p>
    <p>Small icon: <img src="icon.gif" alt="Icon" width="32" height="32"></p>
    """
    result = convert_html_to_markdown(html)
    # Both images should be kept as we don't have size-based filtering in the converter
    assert "![Large]" in result
    assert "![Icon]" in result


def test_image_alt_text():
    """Test that alt text is correctly preserved."""
    html = '<p><img src="test.jpg" alt="Special & <Characters>"></p>'
    result = convert_html_to_markdown(html)
    # Alt text should be properly escaped
    assert "![Special & <Characters>]" in result


def test_logo_removal():
    """Test that logo images are removed during preprocessing."""
    # Create a directory for images to prevent all images from being removed
    test_img_dir = "test_preprocessing_images"
    os.makedirs(test_img_dir, exist_ok=True)

    try:
        html = """
        <p>Content with <img src="logo.png" alt="Logo"></p>
        <p>Content with <img src="not-logo.png" alt="Not Logo"></p>
        """
        # Use specific images_dir to test image processing
        processed = preprocess_html(html, test_img_dir)
        # Logo image should be removed based on filename
        assert 'src="logo.png"' not in processed
        # Text content should be kept
        assert "Content with" in processed
    finally:
        # Clean up
        shutil.rmtree(test_img_dir, ignore_errors=True)


def test_image_path_handling():
    """Test handling of different image path formats."""
    html = """
    <p><img src="relative.jpg" alt="Relative"></p>
    <p><img src="/absolute/path.jpg" alt="Absolute"></p>
    <p><img src="http://example.com/remote.jpg" alt="Remote"></p>
    """
    result = convert_html_to_markdown(html)
    # All image types should be handled appropriately
    assert "![Relative]" in result
    assert "![Absolute]" in result
    assert "![Remote]" in result
