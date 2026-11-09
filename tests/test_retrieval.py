from __future__ import annotations

import asyncio

from sag2 import SearchIndex, search_sag2
from sag2.metrics import ranking_metrics


class KeywordJudge:
    async def score(self, query, candidates, evidence):
        scored = [
            {
                **candidate,
                "jev_score": float(
                    any(word in candidate["content"].casefold() for word in query.split())
                ),
            }
            for candidate in candidates
        ]
        return scored, {
            "requests": 1,
            "judgments": len(candidates),
            "input_tokens": 0,
            "output_tokens": 0,
        }


def test_sag2_returns_the_relevant_document():
    corpus = [
        {"id": "d1", "title": "Cats", "text": "Cats are small domesticated animals."},
        {"id": "d2", "title": "Rockets", "text": "Rockets reach orbit using powerful engines."},
    ]
    index = SearchIndex(corpus, chunk_chars=200)
    try:
        result = asyncio.run(
            search_sag2(
                index,
                KeywordJudge(),
                "Which passage discusses rockets and orbit?",
                candidate_k=2,
                seed_k=1,
                max_hops=0,
            )
        )
    finally:
        index.close()
    assert result["candidates"][0]["external_id"] == "d2"
    assert result["trace"][0]["candidate_count"] >= 1


def test_metrics_use_graded_relevance():
    metrics = ranking_metrics(
        ["missing", "relevant", "partly"],
        {"relevant": 2, "partly": 1},
    )
    assert metrics["mrr@10"] == 0.5
    assert 0.0 < metrics["ndcg@10"] < 1.0
    assert metrics["recall@10"] == 1.0

