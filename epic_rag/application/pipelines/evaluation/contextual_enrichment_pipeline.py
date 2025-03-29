"""ZenML pipeline for evaluating contextual enrichment impact on retrieval quality."""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from zenml import pipeline, step

from ...use_cases.ingest_document import IngestDocumentUseCase
from ...use_cases.retrieve_context import RetrieveContextUseCase


@step
def load_evaluation_dataset(
    dataset_path: str,
) -> Dict[str, Any]:
    """Load an evaluation dataset from a file.

    Args:
        dataset_path: Path to the evaluation dataset file

    Returns:
        Evaluation dataset
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded evaluation dataset with {len(dataset)} queries from {dataset_path}")
    return {"queries": dataset}


@step
def prepare_document_variations(
    document_path: str,
    output_dir: str,
) -> Dict[str, str]:
    """Prepare regular and contextually enriched versions of a document.

    This step prepares two documents:
    1. Original document chunked normally
    2. Same document chunked and contextually enriched

    Args:
        document_path: Path to the source document
        output_dir: Directory to store processing artifacts

    Returns:
        Dictionary with document IDs
    """
    import asyncio
    from ....domain.models.document import Document
    from ....infrastructure.container import container

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read document
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Create document objects with different IDs
    base_document = Document(
        title=Path(document_path).stem,
        content=content,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "source": "evaluation",
            "filename": os.path.basename(document_path),
            "enriched": False,
        },
    )

    enriched_document = Document(
        title=f"{Path(document_path).stem}_enriched",
        content=content,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "source": "evaluation",
            "filename": os.path.basename(document_path),
            "enriched": True,
        },
    )

    # Process the documents
    async def process_documents():
        # Get required services
        document_repository = container.get("document_repository")
        vector_repository = container.get("vector_repository")
        chunking_service = container.get("chunking_service")
        embedding_service = container.get("embedding_service")

        # Try to get the contextual enrichment service
        try:
            container.get("contextual_enrichment_service")
            has_enrichment = True
        except Exception:
            print("Warning: Contextual enrichment service not found in container")
            has_enrichment = False

        # Create use case
        ingest_use_case = IngestDocumentUseCase(
            document_repository=document_repository,
            vector_repository=vector_repository,
            chunking_service=chunking_service,
            embedding_service=embedding_service,
        )

        # Process base document
        print("Processing base document...")
        base_result = await ingest_use_case.execute(
            document=base_document,
            dynamic_chunking=True,
            min_chunk_size=200,
            max_chunk_size=500,
            apply_enrichment=False,
        )

        # Process enriched document
        print("Processing enriched document...")
        if has_enrichment:
            enriched_result = await ingest_use_case.execute(
                document=enriched_document,
                dynamic_chunking=True,
                min_chunk_size=200,
                max_chunk_size=500,
                apply_enrichment=True,
            )
        else:
            print(
                "Warning: No enrichment service available, skipping enriched document"
            )

        return base_document.id, enriched_document.id

    # Run the async function
    base_id, enriched_id = asyncio.run(process_documents())

    print(f"Prepared documents for evaluation:")
    print(f"  Base document ID: {base_id}")
    print(f"  Enriched document ID: {enriched_id}")

    return {
        "base_document_id": base_id,
        "enriched_document_id": enriched_id,
    }


@step
def evaluate_query(
    query: Dict[str, Any],
    document_ids: Dict[str, str],
    first_stage_k: int = 20,
    second_stage_k: int = 5,
    max_results: int = 100,
) -> Dict[str, Any]:
    """Evaluate a single query with both regular and enriched documents.

    Args:
        query: Query data from the evaluation dataset
        document_ids: Dictionary with base and enriched document IDs
        first_stage_k: Number of results for first retrieval stage
        second_stage_k: Number of results for second retrieval stage
        max_results: Maximum number of results to retrieve

    Returns:
        Evaluation results for this query
    """
    import asyncio
    from ....infrastructure.container import container
    from .metrics import calculate_retrieval_metrics

    # Extract query data
    query_text = query["query"]
    relevant_chunk_ids = query.get("relevant_chunk_ids", [])

    async def process_query():
        # Get services
        embedding_service = container.get("embedding_service")
        retrieval_service = container.get("retrieval_service")

        # Create use case
        retrieve_use_case = RetrieveContextUseCase(
            embedding_service=embedding_service,
            retrieval_service=retrieval_service,
        )

        # Process the query against the base document
        base_result = await retrieve_use_case.execute(
            query_text=query_text,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            min_relevance_score=0.0,  # No filtering for evaluation
            use_query_transformation=False,  # Raw query
            merge_related_chunks=False,  # No merging for fair comparison
            filter_metadata={"document_id": document_ids["base_document_id"]},
        )

        # Process the query against the enriched document
        enriched_result = await retrieve_use_case.execute(
            query_text=query_text,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            min_relevance_score=0.0,  # No filtering for evaluation
            use_query_transformation=False,  # Raw query
            merge_related_chunks=False,  # No merging for fair comparison
            filter_metadata={"document_id": document_ids["enriched_document_id"]},
        )

        return base_result, enriched_result

    # Run the async function
    base_result, enriched_result = asyncio.run(process_query())

    # Extract retrieved chunks
    base_retrieved_ids = [chunk.id for chunk in base_result.first_stage_results.chunks]
    enriched_retrieved_ids = [
        chunk.id for chunk in enriched_result.first_stage_results.chunks
    ]

    # Calculate metrics
    base_metrics = calculate_retrieval_metrics(
        retrieved_ids=base_retrieved_ids,
        relevant_ids=relevant_chunk_ids,
    )

    enriched_metrics = calculate_retrieval_metrics(
        retrieved_ids=enriched_retrieved_ids,
        relevant_ids=relevant_chunk_ids,
    )

    # Prepare result
    result = {
        "query": query_text,
        "relevant_chunks": relevant_chunk_ids,
        "base_metrics": base_metrics.as_dict(),
        "enriched_metrics": enriched_metrics.as_dict(),
        "base_retrieved": base_retrieved_ids[:10],  # Include first 10 for inspection
        "enriched_retrieved": enriched_retrieved_ids[
            :10
        ],  # Include first 10 for inspection
        "base_latency_ms": base_result.total_latency_ms,
        "enriched_latency_ms": enriched_result.total_latency_ms,
    }

    # Calculate improvement
    if len(relevant_chunk_ids) > 0:
        base_recall_20 = base_metrics.recall_at_k.get(20, 0.0)
        enriched_recall_20 = enriched_metrics.recall_at_k.get(20, 0.0)

        # Anthropic's metric: failure rate reduction at 20
        # 1 - recall@20 = failure rate
        base_failure_rate = 1 - base_recall_20
        enriched_failure_rate = 1 - enriched_recall_20

        if base_failure_rate > 0:
            failure_rate_reduction = (
                base_failure_rate - enriched_failure_rate
            ) / base_failure_rate
        else:
            failure_rate_reduction = 0.0

        result["improvement"] = {
            "recall_at_20_absolute": enriched_recall_20 - base_recall_20,
            "recall_at_20_relative": (
                (enriched_recall_20 / base_recall_20) - 1
                if base_recall_20 > 0
                else float("inf")
            ),
            "failure_rate_reduction": failure_rate_reduction,
        }

    return result


@step
def analyze_evaluation_results(
    queries: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    output_path: str,
) -> Dict[str, Any]:
    """Analyze the evaluation results and generate a report.

    Args:
        queries: List of query data
        results: List of evaluation results
        output_path: Path to save the report

    Returns:
        Summary metrics
    """
    import os
    import json
    from .metrics import RetrievalMetrics, calculate_aggregate_metrics

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert individual metrics to RetrievalMetrics objects
    base_metrics_list = []
    enriched_metrics_list = []

    for result in results:
        # Base metrics
        base_recall = result["base_metrics"]["recall_at_k"]
        base_precision = result["base_metrics"]["precision_at_k"]
        base_ndcg = result["base_metrics"]["ndcg_at_k"]
        base_mrr = result["base_metrics"]["mean_reciprocal_rank"]

        # Convert string keys to integers
        base_recall = {int(k): v for k, v in base_recall.items()}
        base_precision = {int(k): v for k, v in base_precision.items()}
        base_ndcg = {int(k): v for k, v in base_ndcg.items()}

        base_metrics = RetrievalMetrics(
            recall_at_k=base_recall,
            precision_at_k=base_precision,
            ndcg_at_k=base_ndcg,
            mean_reciprocal_rank=base_mrr,
        )
        base_metrics_list.append(base_metrics)

        # Enriched metrics
        enriched_recall = result["enriched_metrics"]["recall_at_k"]
        enriched_precision = result["enriched_metrics"]["precision_at_k"]
        enriched_ndcg = result["enriched_metrics"]["ndcg_at_k"]
        enriched_mrr = result["enriched_metrics"]["mean_reciprocal_rank"]

        # Convert string keys to integers
        enriched_recall = {int(k): v for k, v in enriched_recall.items()}
        enriched_precision = {int(k): v for k, v in enriched_precision.items()}
        enriched_ndcg = {int(k): v for k, v in enriched_ndcg.items()}

        enriched_metrics = RetrievalMetrics(
            recall_at_k=enriched_recall,
            precision_at_k=enriched_precision,
            ndcg_at_k=enriched_ndcg,
            mean_reciprocal_rank=enriched_mrr,
        )
        enriched_metrics_list.append(enriched_metrics)

    # Calculate aggregate metrics
    aggregate_base = calculate_aggregate_metrics(base_metrics_list)
    aggregate_enriched = calculate_aggregate_metrics(enriched_metrics_list)

    # Calculate improvement metrics
    improvements = {}
    for k in sorted(aggregate_base.recall_at_k.keys()):
        base_val = aggregate_base.recall_at_k.get(k, 0.0)
        enriched_val = aggregate_enriched.recall_at_k.get(k, 0.0)
        abs_diff = enriched_val - base_val
        rel_diff = (enriched_val / base_val) - 1 if base_val > 0 else float("inf")

        improvements[f"recall@{k}"] = {
            "base": base_val,
            "enriched": enriched_val,
            "absolute_improvement": abs_diff,
            "relative_improvement": rel_diff,
        }

    # Calculate Anthropic's metric: failure rate reduction at 20
    base_failure_rate = 1 - aggregate_base.recall_at_k.get(20, 0.0)
    enriched_failure_rate = 1 - aggregate_enriched.recall_at_k.get(20, 0.0)

    if base_failure_rate > 0:
        failure_rate_reduction = (
            base_failure_rate - enriched_failure_rate
        ) / base_failure_rate
    else:
        failure_rate_reduction = 0.0

    # Calculate average latency
    avg_base_latency = sum(r["base_latency_ms"] for r in results) / len(results)
    avg_enriched_latency = sum(r["enriched_latency_ms"] for r in results) / len(results)

    # Prepare summary
    summary = {
        "total_queries": len(queries),
        "base_metrics": aggregate_base.as_dict(),
        "enriched_metrics": aggregate_enriched.as_dict(),
        "improvements": improvements,
        "anthropic_metric": {
            "base_failure_rate": base_failure_rate,
            "enriched_failure_rate": enriched_failure_rate,
            "failure_rate_reduction": failure_rate_reduction,
        },
        "latency": {
            "base_avg_ms": avg_base_latency,
            "enriched_avg_ms": avg_enriched_latency,
            "overhead_ms": avg_enriched_latency - avg_base_latency,
            "overhead_percent": (
                ((avg_enriched_latency / avg_base_latency) - 1) * 100
                if avg_base_latency > 0
                else 0
            ),
        },
    }

    # Save detailed results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "individual_results": results,
            },
            f,
            indent=2,
        )

    print(f"Evaluation complete. Report saved to {output_path}")
    print("\nSummary:")
    print(f"Queries: {len(queries)}")
    print(f"Base Recall@20: {aggregate_base.recall_at_k.get(20, 0):.4f}")
    print(f"Enriched Recall@20: {aggregate_enriched.recall_at_k.get(20, 0):.4f}")
    print(f"Failure Rate Reduction: {failure_rate_reduction:.2%}")
    print(
        f"Latency Overhead: {avg_enriched_latency - avg_base_latency:.2f}ms ({((avg_enriched_latency / avg_base_latency) - 1) * 100:.1f}%)"
    )

    return summary


@pipeline(enable_cache=False) # type: ignore
def contextual_enrichment_evaluation_pipeline(
    dataset_path: str,
    document_path: str,
    output_dir: str = "data/evaluation",
    first_stage_k: int = 20,
    second_stage_k: int = 5,
    max_results: int = 100,
) -> Dict[str, Any]:
    """Pipeline for evaluating the impact of contextual enrichment on retrieval.

    Args:
        dataset_path: Path to the evaluation dataset
        document_path: Path to the document to evaluate
        output_dir: Directory to save evaluation results
        first_stage_k: Number of results for first retrieval stage
        second_stage_k: Number of results for second retrieval stage
        max_results: Maximum number of results to retrieve

    Returns:
        Evaluation summary
    """
    # Step 1: Load evaluation dataset
    dataset = load_evaluation_dataset(dataset_path=dataset_path)

    # Step 2: Prepare document variations
    documents = prepare_document_variations(
        document_path=document_path,
        output_dir=output_dir,
    )

    # Step 3: Evaluate each query
    results = []
    for query in dataset["queries"]:
        result = evaluate_query(
            query=query,
            document_ids=documents,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            max_results=max_results,
        )
        results.append(result)

    # Step 4: Analyze results
    output_path = os.path.join(output_dir, "evaluation_results.json")
    summary = analyze_evaluation_results(
        queries=dataset["queries"],
        results=results,
        output_path=output_path,
    )

    return summary
