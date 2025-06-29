"""Evaluation metrics for RAG systems."""

from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval performance."""

    recall_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    precision_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]

    def __str__(self) -> str:
        """Format metrics as a string."""
        lines = [
            f"Recall@5: {self.recall_at_k.get(5, 0):.4f}",
            f"Recall@10: {self.recall_at_k.get(10, 0):.4f}",
            f"Recall@20: {self.recall_at_k.get(20, 0):.4f}",
            f"MRR: {self.mean_reciprocal_rank:.4f}",
            f"Precision@5: {self.precision_at_k.get(5, 0):.4f}",
            f"NDCG@5: {self.ndcg_at_k.get(5, 0):.4f}",
            f"NDCG@10: {self.ndcg_at_k.get(10, 0):.4f}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "recall_at_k": self.recall_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "precision_at_k": self.precision_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "failure_rate_at_20": 1 - self.recall_at_k.get(20, 0),  # Anthropic metric
        }


def calculate_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = [5, 10, 20, 50, 100],
) -> RetrievalMetrics:
    """Calculate retrieval metrics for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs, in order of relevance
        relevant_ids: List of known relevant document IDs
        k_values: Values of k to compute Recall@k and Precision@k

    Returns:
        RetrievalMetrics object with all metrics
    """
    # Convert to sets for easier operations
    relevant_set = set(relevant_ids)

    # Calculate Recall@k: proportion of relevant documents retrieved at cutoff k
    recall_at_k = {}
    for k in k_values:
        if k <= len(retrieved_ids):
            retrieved_at_k = set(retrieved_ids[:k])
            if len(relevant_set) > 0:
                recall_at_k[k] = len(retrieved_at_k.intersection(relevant_set)) / len(
                    relevant_set
                )
            else:
                recall_at_k[k] = 0.0
        else:
            recall_at_k[k] = recall_at_k.get(min(k_values), 0.0)

    # Calculate Precision@k: proportion of retrieved documents that are relevant at cutoff k
    precision_at_k = {}
    for k in k_values:
        if k <= len(retrieved_ids):
            retrieved_at_k = set(retrieved_ids[:k])
            if k > 0:
                precision_at_k[k] = len(retrieved_at_k.intersection(relevant_set)) / k
            else:
                precision_at_k[k] = 0.0
        else:
            precision_at_k[k] = precision_at_k.get(min(k_values), 0.0)

    # Calculate Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            mrr = 1.0 / (i + 1)
            break

    # Calculate NDCG@k (Normalized Discounted Cumulative Gain)
    ndcg_at_k = {}
    for k in k_values:
        if k <= len(retrieved_ids):
            # Calculate DCG - all relevant docs have relevance of 1.0
            dcg = 0.0
            for i, doc_id in enumerate(retrieved_ids[:k]):
                if doc_id in relevant_set:
                    # Using log base 2 for standard NDCG calculation
                    dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

            # Calculate ideal DCG (IDCG)
            idcg = 0.0
            for i in range(min(len(relevant_set), k)):
                idcg += 1.0 / np.log2(i + 2)

            # Calculate NDCG
            ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0
        else:
            ndcg_at_k[k] = ndcg_at_k.get(min(k_values), 0.0)

    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        mean_reciprocal_rank=mrr,
        precision_at_k=precision_at_k,
        ndcg_at_k=ndcg_at_k,
    )


def calculate_aggregate_metrics(
    metrics_list: List[RetrievalMetrics],
) -> RetrievalMetrics:
    """Calculate aggregate metrics across multiple queries.

    Args:
        metrics_list: List of RetrievalMetrics objects

    Returns:
        RetrievalMetrics object with averaged metrics
    """
    if not metrics_list:
        return RetrievalMetrics(
            recall_at_k={},
            mean_reciprocal_rank=0.0,
            precision_at_k={},
            ndcg_at_k={},
        )

    # Collect all k values
    k_values = set()
    for metrics in metrics_list:
        k_values.update(metrics.recall_at_k.keys())

    # Aggregate Recall@k
    agg_recall_at_k = {}
    for k in k_values:
        values = [m.recall_at_k.get(k, 0.0) for m in metrics_list]
        agg_recall_at_k[k] = sum(values) / len(metrics_list)

    # Aggregate Precision@k
    agg_precision_at_k = {}
    for k in k_values:
        values = [m.precision_at_k.get(k, 0.0) for m in metrics_list]
        agg_precision_at_k[k] = sum(values) / len(metrics_list)

    # Aggregate MRR
    agg_mrr = sum(m.mean_reciprocal_rank for m in metrics_list) / len(metrics_list)

    # Aggregate NDCG@k
    agg_ndcg_at_k = {}
    for k in k_values:
        values = [m.ndcg_at_k.get(k, 0.0) for m in metrics_list]
        agg_ndcg_at_k[k] = sum(values) / len(metrics_list)

    return RetrievalMetrics(
        recall_at_k=agg_recall_at_k,
        mean_reciprocal_rank=agg_mrr,
        precision_at_k=agg_precision_at_k,
        ndcg_at_k=agg_ndcg_at_k,
    )
