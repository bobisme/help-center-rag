"""Core functionality for HTML to Markdown conversion."""

import os
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify


def _convert_spans_to_strong(soup: BeautifulSoup) -> None:
    """Convert spans with font-weight: bold to <strong> tags."""
    for span in soup.find_all("span"):
        if not isinstance(span, Tag):
            continue

        # Check if it has a style attribute with font-weight: bold
        style = span.get("style", "")
        if not isinstance(style, str) or "font-weight: bold" not in style:
            continue

        # Check if it's already wrapped in a strong tag
        parent = span.parent
        if parent and parent.name == "strong":
            continue

        # Create a new strong tag
        strong_tag = soup.new_tag("strong")

        # Move the contents to the strong tag
        for content in list(span.contents):
            strong_tag.append(content.extract())

        # Replace the span with the strong tag
        span.replace_with(strong_tag)


def _convert_spans_to_em(soup: BeautifulSoup) -> None:
    """Convert spans with font-style: italic to <em> tags."""
    for span in soup.find_all("span"):
        if not isinstance(span, Tag):
            continue

        # Check if it has a style attribute with font-style: italic
        style = span.get("style", "")
        if not isinstance(style, str) or "font-style: italic" not in style:
            continue

        # Skip if the style also contains font-style: normal
        if "font-style: normal" in style:
            continue

        # Check if it's already wrapped in an em tag
        parent = span.parent
        if parent and parent.name == "em":
            continue

        # Create a new em tag
        em_tag = soup.new_tag("em")

        # Move the contents to the em tag
        for content in list(span.contents):
            em_tag.append(content.extract())

        # Replace the span with the em tag
        span.replace_with(em_tag)


def _convert_inline_styles_to_semantic_tags(soup: BeautifulSoup) -> None:
    """
    Convert inline style attributes for bold and italic text to semantic tags.

    Converts spans with inline styles to <strong> and <em> tags.
    This improves the markdown conversion for styled text.
    """
    _convert_spans_to_strong(soup)
    _convert_spans_to_em(soup)


def _fix_nested_lists(soup: BeautifulSoup) -> None:
    """Fix nested list structure that uses empty list items as containers."""
    for list_item in soup.find_all("li"):
        if not isinstance(list_item, Tag) or not list_item.get("style"):
            continue

        # Check if this is a list item with style="list-style: none"
        if "list-style: none" not in list_item["style"]:
            continue

        # Check if this is a list item containing a nested list
        inner_list = list_item.find(["ul", "ol"])
        if not inner_list or not isinstance(inner_list, Tag):
            continue

        # Find the previous list item, which should be the parent for list
        prev_li = list_item.find_previous_sibling("li")
        if prev_li and isinstance(prev_li, Tag):
            # Move the inner list inside the previous list item
            inner_list.extract()  # Remove from current position
            prev_li.append(inner_list)  # Add to the previous list item
            # Remove the empty list item
            list_item.decompose()


def _process_links(soup: BeautifulSoup) -> None:
    """Replace javascript and local file links with just their text content."""
    for link in soup.find_all("a"):
        # Make sure we're dealing with a Tag, not NavigableString
        if not isinstance(link, Tag):
            continue

        # If the link is javascript or a local file reference, replace with text
        href = link.get("href", "")
        is_local_link = isinstance(href, str) and (
            href.startswith("javascript:")
            or href.endswith(".htm")
            or href.startswith("../")
        )
        if is_local_link:
            # Preserve the inner text
            link_text = link.get_text(strip=True)
            # Create a new string element to replace the link
            new_element = soup.new_string(link_text)
            link.replace_with(new_element)


def _update_image_src(img: Tag, images_dir: str, filename: str) -> bool:
    """
    Update image src attribute to point to a local file.

    Returns True if a matching file was found, False otherwise.
    """
    # First try exact filename matches or contains
    for img_file in os.listdir(images_dir):
        if img_file.endswith(filename) or filename in img_file:
            img["src"] = f"{images_dir}/{img_file}"
            return True

    # If not found, try matching parts of the filename
    for img_file in os.listdir(images_dir):
        if any(part in img_file.lower() for part in filename.lower().split(".")):
            img["src"] = f"{images_dir}/{img_file}"
            return True

    return False


def _process_images(soup: BeautifulSoup, images_dir: Optional[str]) -> None:
    """Process images to point to local files or remove them if needed."""
    for img in soup.find_all("img"):
        # Make sure we're dealing with a Tag, not NavigableString
        if not isinstance(img, Tag):
            continue

        src = img.get("src", "")
        if not isinstance(src, str) or not src or src.startswith("data:"):
            continue

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
            # Try to find a matching file in the images directory
            if not _update_image_src(img, images_dir, filename):
                # If no match found and not an external URL, remove the image
                if not src.startswith("http"):
                    img.decompose()


def _remove_header_elements(soup: BeautifulSoup) -> None:
    """Remove logo image and header text elements."""
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


def preprocess_html(html: str, images_dir: Optional[str] = None) -> str:
    """
    Process the HTML before conversion to markdown.

    This includes:
    - Removing header elements and context links
    - Fixing nested list structure
    - Converting inline styles to semantic tags
    - Processing links and images

    Args:
        html: The HTML content to preprocess
        images_dir: Optional directory path containing images for local references

    Returns:
        Preprocessed HTML ready for markdown conversion
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove header elements
    _remove_header_elements(soup)

    # Fix nested list structure
    _fix_nested_lists(soup)

    # Convert inline styles for bold and italic to proper semantic tags
    _convert_inline_styles_to_semantic_tags(soup)

    # Process links
    _process_links(soup)

    # Process images
    _process_images(soup, images_dir)

    return str(soup)


def convert_html_to_markdown(
    html: str,
    images_dir: Optional[str] = None,
    heading_style: str = "ATX",
    wrap: bool = True,
) -> str:
    """
    Convert HTML to Markdown with preprocessing for Epic documentation.

    Args:
        html: The HTML content to convert
        images_dir: Optional directory path containing images for local references
        heading_style: The style for headings ('ATX' or 'SETEXT')
        wrap: Whether to wrap long lines

    Returns:
        Markdown formatted text
    """
    # Preprocess the HTML
    processed_html = preprocess_html(html, images_dir)

    # Convert to markdown
    return markdownify(processed_html, heading_style=heading_style, wrap=wrap)
