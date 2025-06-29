"""Step to convert HTML to Markdown."""

import json
import os
from typing import Optional, List

from zenml import step

from html2md import convert_html_to_markdown, preprocess_html
from help_rag.domain.models.document import Document


@step
def convert_to_markdown(
    index: Optional[int] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    all_docs: bool = True,
    source_path: str = "output/scraped-docs.json",
    images_dir: str = "output/images",
) -> List[Document]:
    """Convert HTML content to Markdown format.

    If no parameters are specified, this processes all documents by default.

    Args:
        index: Optional specific index of the document to convert
        offset: Starting index for batch processing
        limit: Maximum number of documents to process
        all_docs: Whether to process all documents
        source_path: Path to the source JSON file
        images_dir: Directory where images are stored

    Returns:
        List of Document objects with markdown content
    """
    if not os.path.exists(source_path):
        raise ValueError(f"File not found: {source_path}")

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "pages" not in data or not isinstance(data["pages"], list):
            raise ValueError(
                f"Invalid format: {source_path} is not a consolidated file"
            )

        total_pages = len(data["pages"])
        print(f"Found {total_pages} pages in {source_path}")

        # Determine which pages to process
        if index is not None:
            if index >= total_pages:
                raise ValueError(f"Index out of range: {index} (max: {total_pages-1})")
            pages_to_process = [index]
        elif all_docs:
            pages_to_process = range(total_pages)
        else:
            # Apply offset and limit
            start_idx = min(offset, total_pages)
            if limit is not None:
                end_idx = min(start_idx + limit, total_pages)
            else:
                end_idx = total_pages
            pages_to_process = range(start_idx, end_idx)

        documents = []
        images_dir_checked = False

        for idx in pages_to_process:
            page = data["pages"][idx]

            # Extract document data
            title = page.get("title", f"Untitled_Page_{idx}")
            page_id = idx  # Use index as ID
            category = page.get("metadata", {}).get("path", ["Uncategorized"])[0]
            updated_at = page.get("metadata", {}).get("crawlDate")

            # Convert HTML content to markdown
            content = page.get("content", "")
            if not content and "rawHtml" in page:
                # Only check images directory once
                if not images_dir_checked:
                    if os.path.exists(images_dir):
                        print(f"Using images from {images_dir}")
                    else:
                        print(
                            f"Warning: Images directory {images_dir} not found. Images will be removed."
                        )
                    images_dir_checked = True

                html = preprocess_html(page["rawHtml"], images_dir)
                content = convert_html_to_markdown(html, images_dir=images_dir)

            # Check if content already starts with the title as a heading
            has_title_heading = False
            if content:
                heading_pattern = f"# {title}"
                has_title_heading = content.strip().startswith(heading_pattern)

            # Add title heading only if not already present
            final_content = content
            if not has_title_heading:
                final_content = f"# {title}\n\n{content}"

            # Create document
            document = Document(
                title=title,
                content=final_content,
                source_page_id=str(page_id) if page_id is not None else None,
                metadata={
                    "category": category,
                    "updated_at": updated_at,
                    "source_path": source_path,
                    "page_index": idx,
                },
            )

            documents.append(document)

        print(f"Converted {len(documents)} documents to markdown")
        return documents

    except Exception as e:
        raise Exception(f"Error converting to markdown: {str(e)}")
