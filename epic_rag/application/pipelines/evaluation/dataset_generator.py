"""Dataset generator for RAG evaluation."""

import os
import json
from datetime import datetime
from pathlib import Path

from ....domain.models.document import Document
from ....infrastructure.container import container


async def generate_evaluation_dataset(
    input_file_path: str,
    output_path: str,
    num_queries: int = 20,
    num_relevant_per_query: int = 3,
) -> str:
    """Generate an evaluation dataset from a document.

    Args:
        input_file_path: Path to the markdown document to use
        output_path: Directory to save the dataset
        num_queries: Number of evaluation queries to generate
        num_relevant_per_query: Number of relevant chunks per query

    Returns:
        Path to the generated dataset
    """
    print(f"Generating evaluation dataset from {input_file_path}...")

    # Make sure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Read the input document
    with open(input_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Get services using type-based dependency injection
    from ....domain.services.chunking_service import ChunkingService

    chunking_service = container[ChunkingService]

    # Create a document
    document = Document(
        title=Path(input_file_path).stem,
        content=content,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "source": "evaluation",
            "filename": os.path.basename(input_file_path),
        },
    )

    # Process the document with chunking service
    chunks = await chunking_service.dynamic_chunk_document(
        document=document,
        min_chunk_size=200,
        max_chunk_size=500,
    )

    print(f"Document chunked into {len(chunks)} chunks")

    # Now use LLM to generate queries and relevant chunks
    from ....domain.services.llm_service import LLMService

    llm_service = container[LLMService]

    # Format chunks for the prompt
    chunks_text = "\n\n".join(
        [
            f"CHUNK {i+1}:\n{chunk.content}"
            for i, chunk in enumerate(
                chunks[:50]
            )  # Limit to first 50 chunks to avoid context limits
        ]
    )

    # Create the prompt
    prompt = f"""
You are an expert in creating evaluation datasets for retrieval systems.

I have a document that has been split into chunks. I need you to generate {num_queries} realistic queries 
that a user might ask about this content and identify which chunks contain relevant information for each query.

For each query:
1. Create a natural, specific question about the content
2. Identify {num_relevant_per_query} relevant chunk numbers that contain information needed to answer the query
3. Explain why each chunk is relevant to the query

Here are the document chunks:

{chunks_text}

Please format your response as a JSON array where each element is an object with these fields:
- "query": The user query
- "relevant_chunks": Array of relevant chunk numbers (1-based indexing)
- "explanation": Brief explanation of why these chunks are relevant

ONLY return the JSON array, no introduction or other text.
"""

    # Get response from LLM
    response = await llm_service.generate_text(prompt)

    try:
        # Clean up the response to handle markdown code blocks
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]  # Remove ```json prefix
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]  # Remove ``` suffix

        # Parse the JSON response
        dataset = json.loads(clean_response)

        # Add the actual chunk content
        for item in dataset:
            # Convert 1-indexed to 0-indexed
            chunk_indices = [int(idx) - 1 for idx in item["relevant_chunks"]]
            item["relevant_content"] = [
                chunks[idx].content if idx < len(chunks) else "CHUNK INDEX OUT OF RANGE"
                for idx in chunk_indices
            ]
            # Add chunk IDs
            item["relevant_chunk_ids"] = [
                chunks[idx].id if idx < len(chunks) else "INVALID"
                for idx in chunk_indices
            ]

        # Save the dataset
        output_file = os.path.join(output_path, "evaluation_dataset.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        print(
            f"Created evaluation dataset with {len(dataset)} queries at {output_file}"
        )
        return output_file

    except json.JSONDecodeError:
        error_output = os.path.join(output_path, "llm_response_error.txt")
        with open(error_output, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Error parsing LLM response. Raw output saved to {error_output}")
        return error_output
