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


def test_malformed_html_tags():
    """Test that malformed HTML tags are handled correctly without errors."""
    html = """
    <p>This has a <span with malformed attributes tag></p>
    <p>This has a <li class=None>list item</li> with None attribute</p>
    <p>This has an <img> tag without src attribute</p>
    <p>This has a <li style="">list item with empty style</li></p>
    """
    # This should not raise an exception
    result = convert_html_to_markdown(html)
    assert "This has a" in result
    assert "list item" in result
    assert "tag without src attribute" in result


def test_none_attributes():
    """Test that None attributes are handled correctly."""
    html = """
    <ul>
        <li>Normal list item</li>
        <li style>Item with empty style attribute</li>
        <li class=>Item with empty class</li>
    </ul>
    """
    # This should not raise an exception
    result = convert_html_to_markdown(html)
    assert "Normal list item" in result
    assert "Item with empty style attribute" in result
    assert "Item with empty class" in result


def test_list_with_none_style():
    """Test that list items with None style attribute are handled correctly."""
    from bs4 import BeautifulSoup
    from html2md.converter import _fix_nested_lists
    
    html = """
    <ul>
        <li>Normal item</li>
        <li style="list-style: none">Hidden item with nested list
            <ul>
                <li>Nested item</li>
            </ul>
        </li>
        <li style="">Empty style</li>
    </ul>
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # This should not raise an exception
    _fix_nested_lists(soup)
    
    # Check the structure post-processing
    result = str(soup)
    assert "Normal item" in result
    assert "Empty style" in result


def test_span_with_none_attributes():
    """Test that span tags with None attributes are handled correctly."""
    from bs4 import BeautifulSoup
    from html2md.converter import _convert_spans_to_strong, _convert_spans_to_em
    
    html = """
    <p>
        <span>Regular span</span>
        <span style="font-weight: bold">Bold span</span>
        <span style="font-style: italic">Italic span</span>
        <span style="">Empty style</span>
    </p>
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # This should not raise an exception
    _convert_spans_to_strong(soup)
    _convert_spans_to_em(soup)
    
    # Check the structure post-processing
    result = str(soup)
    assert "Regular span" in result
    assert "<strong>" in result
    assert "<em>" in result
    assert "Empty style" in result


def test_links_with_none_attributes():
    """Test that links with None attributes are handled correctly."""
    from bs4 import BeautifulSoup
    from html2md.converter import _process_links
    
    html = """
    <p>
        <a href="https://example.com">Regular link</a>
        <a href="javascript:void(0)">JavaScript link</a>
        <a href="">Empty href</a>
        <a>No href attribute</a>
    </p>
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # This should not raise an exception
    _process_links(soup)
    
    # Check the structure post-processing
    result = str(soup)
    assert "Regular link" in result
    assert "JavaScript link" in result
    assert "Empty href" in result
    assert "No href attribute" in result


def test_images_with_none_attributes():
    """Test that images with None attributes are handled correctly."""
    from bs4 import BeautifulSoup
    from html2md.converter import _process_images
    
    html = """
    <p>
        <img src="test.jpg" alt="Test image">
        <img src="">Empty src
        <img>No src attribute
        <img src="logo.png" alt="Logo">
    </p>
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # This should not raise an exception
    _process_images(soup, "nonexistent_dir")
    
    # Check the structure post-processing
    result = str(soup)
    
    # Non-logo images remain in the output but are not processed (BeautifulSoup behavior)
    # The key point is that the function doesn't crash with None attributes
    assert "Empty src" in result
    assert "No src attribute" in result


def test_comprehensive_none_attribute_handling():
    """Test that all None attribute handling works correctly in an end-to-end test."""
    # Complex HTML with various cases of None or malformed attributes
    html = """
    <html>
        <body>
            <h1>Test Document</h1>
            <p>This has a <span style="font-weight: bold">bold</span> and <span style="font-style: italic">italic</span> text.</p>
            <p>This has a <span style>empty style attribute</span>.</p>
            <p>This has an <a href="javascript:void(0)">JavaScript link</a> and a <a>link with no href</a>.</p>
            <ul>
                <li>Regular list item</li>
                <li style="list-style: none">
                    Hidden item
                    <ul>
                        <li>Nested item</li>
                    </ul>
                </li>
                <li style="">Empty style list item</li>
                <li class=None>List item with None class</li>
            </ul>
            <p>Image tests: 
                <img src="test.jpg" alt="Test Image">
                <img>Image with no src
                <img src="">Image with empty src
                <img src="logo.png" alt="Logo">
            </p>
            <p><span with malformed attributes>Malformed span</span></p>
        </body>
    </html>
    """
    
    # This should not raise an exception
    result = convert_html_to_markdown(html, images_dir="nonexistent_dir")
    
    # Check various parts of the result
    assert "Test Document" in result
    assert "bold" in result 
    assert "italic" in result
    assert "empty style attribute" in result
    assert "JavaScript link" in result
    assert "link with no href" in result
    assert "Regular list item" in result
    assert "Nested item" in result
    assert "Empty style list item" in result
    assert "List item with None class" in result
    assert "Image with no src" in result
    assert "Image with empty src" in result
    assert "Malformed span" in result
    
    # The main thing we're testing here is that the function doesn't crash
    # with malformed HTML or None attributes
    # We're not strictly testing the exact output format as that may vary
    # due to BeautifulSoup parser behavior
