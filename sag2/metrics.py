"""Ranking metrics shared by the multi-hop dataset experiments."""

from __future__ import annotations

import math
from typing import Any


def dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def ranking_metrics(
    ranked_ids: list[str], relevance: dict[str, int], cutoff: int = 10
) -> dict[str, float]:
    ranked = ranked_ids[:cutoff]
    gains = [relevance.get(doc_id, 0) for doc_id in ranked]
    ideal = sorted(relevance.values(), reverse=True)[:cutoff]
    ideal_gain = dcg(ideal)
    first_relevant = next(
        (rank for rank, doc_id in enumerate(ranked, 1) if relevance.get(doc_id, 0) > 0),
        None,
    )
    return {
        f"ndcg@{cutoff}": dcg(gains) / ideal_gain if ideal_gain else 0.0,
        "recall@2": sum(
            relevance.get(doc_id, 0) > 0 for doc_id in ranked[:2]
        ) / min(len(relevance), 2),
        f"recall@{cutoff}": sum(
            relevance.get(doc_id, 0) > 0 for doc_id in ranked
        ) / min(len(relevance), cutoff),
        f"mrr@{cutoff}": 1.0 / first_relevant if first_relevant else 0.0,
    }


def unique_documents(hits: list[dict[str, Any]], cutoff: int = 10) -> list[str]:
    result = []
    seen = set()
    for hit in hits:
        doc_id = str(hit["external_id"])
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
        if len(result) == cutoff:
            break
    return result


def mean_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, float]:
    keys = ("ndcg@10", "recall@2", "recall@10", "mrr@10")
    return (
        {
            key: sum(float(row[system][key]) for row in rows) / len(rows)
            for key in keys
        }
        if rows
        else {key: 0.0 for key in keys}
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * quantile)]

