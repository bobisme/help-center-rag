"""ZenML pipeline for evaluating query processing with the RAG system."""
from typing import List, Dict, Any, Optional

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.steps import Output

from ...domain.models.retrieval import Query, ContextualRetrievalResult
from ...infrastructure.container import container
from ..use_cases.retrieve_context import RetrieveContextUseCase


@step
def load_test_queries(
    query_file: str = "data/test_queries.txt",
) -> List[str]:
    """Load test queries from a file.
    
    Args:
        query_file: Path to the file containing test queries
        
    Returns:
        List of query strings
    """
    # Read queries from file (one per line)
    with open(query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(queries)} test queries from {query_file}")
    return queries


@step
def process_query(
    query_text: str,
    first_stage_k: int = 20,
    second_stage_k: int = 5,
    min_relevance_score: float = 0.7,
    use_query_transformation: bool = True,
    merge_related_chunks: bool = True
) -> Dict[str, Any]:
    """Process a single query and retrieve context.
    
    Args:
        query_text: The query text
        first_stage_k: Number of documents to retrieve in first stage
        second_stage_k: Number of documents to retrieve in second stage
        min_relevance_score: Minimum relevance score for filtering
        use_query_transformation: Whether to transform the query
        merge_related_chunks: Whether to merge related chunks
        
    Returns:
        Dictionary with query results and metrics
    """
    import asyncio
    from ...infrastructure.container import container
    
    # Get required services
    embedding_service = container.get("embedding_service")
    retrieval_service = container.get("retrieval_service")
    
    # Create use case
    use_case = RetrieveContextUseCase(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service
    )
    
    # Process the query
    async def process():
        return await use_case.execute(
            query_text=query_text,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            min_relevance_score=min_relevance_score,
            use_query_transformation=use_query_transformation,
            merge_related_chunks=merge_related_chunks
        )
    
    result = asyncio.run(process())
    
    # Extract metrics and results
    metrics = {
        "query": query_text,
        "total_latency_ms": result.total_latency_ms,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "processing_latency_ms": result.processing_latency_ms,
        "first_stage_count": len(result.first_stage_results.chunks),
        "second_stage_count": len(result.second_stage_results.chunks) if result.second_stage_results else 0,
        "final_chunk_count": len(result.final_chunks),
        "has_results": len(result.final_chunks) > 0,
    }
    
    # Add the actual results
    chunks_data = []
    for chunk in result.final_chunks:
        chunks_data.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "relevance_score": chunk.relevance_score,
            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
        })
    
    return {
        "metrics": metrics,
        "chunks": chunks_data,
        "merged_content": result.merged_content,
    }


@step
def evaluate_all_queries(
    queries: List[str],
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate the results of all queries.
    
    Args:
        queries: List of query strings
        results: List of query results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate overall metrics
    total_queries = len(queries)
    queries_with_results = sum(1 for r in results if r["metrics"]["has_results"])
    
    avg_latency = sum(r["metrics"]["total_latency_ms"] for r in results) / total_queries if total_queries else 0
    avg_first_stage = sum(r["metrics"]["first_stage_count"] for r in results) / total_queries if total_queries else 0
    avg_final_chunks = sum(r["metrics"]["final_chunk_count"] for r in results) / total_queries if total_queries else 0
    
    return {
        "total_queries": total_queries,
        "queries_with_results": queries_with_results,
        "success_rate": queries_with_results / total_queries if total_queries else 0,
        "avg_latency_ms": avg_latency,
        "avg_first_stage_results": avg_first_stage,
        "avg_final_chunks": avg_final_chunks,
        "query_metrics": [r["metrics"] for r in results],
    }


@pipeline(settings={"docker": DockerSettings(required_integrations=["qdrant"])})
def query_evaluation_pipeline(
    query_file: str = "data/test_queries.txt",
    first_stage_k: int = 20,
    second_stage_k: int = 5,
    min_relevance_score: float = 0.7,
    use_query_transformation: bool = True,
    merge_related_chunks: bool = True
) -> Dict[str, Any]:
    """Pipeline for evaluating query processing.
    
    Args:
        query_file: Path to the file containing test queries
        first_stage_k: Number of documents to retrieve in first stage
        second_stage_k: Number of documents to retrieve in second stage
        min_relevance_score: Minimum relevance score for filtering
        use_query_transformation: Whether to transform the query
        merge_related_chunks: Whether to merge related chunks
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Step 1: Load test queries
    queries = load_test_queries(query_file=query_file)
    
    # Step 2: Process each query
    results = []
    for query in queries:
        result = process_query(
            query_text=query,
            first_stage_k=first_stage_k,
            second_stage_k=second_stage_k,
            min_relevance_score=min_relevance_score,
            use_query_transformation=use_query_transformation,
            merge_related_chunks=merge_related_chunks
        )
        results.append(result)
    
    # Step 3: Evaluate results
    evaluation = evaluate_all_queries(queries=queries, results=results)
    
    return evaluation