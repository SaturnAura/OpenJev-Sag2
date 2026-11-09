"""CLI: run one SAG2 query with an in-process OpenJev judge."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from openjev.runtime import load_engine
from sag2 import LocalOpenJeVJudge, SearchConfig, SearchIndex, search_sag2

ROOT = Path(__file__).resolve().parents[1]


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(documents, list) or not documents:
        raise ValueError("corpus must be a non-empty JSON list")
    return [
        {
            **row,
            "title": str(row.get("title") or ""),
            "text": str(row.get("text") or ""),
        }
        for row in documents
        if isinstance(row, dict)
    ]


async def run(args) -> dict[str, Any]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    search = SearchConfig(**{
        key: value
        for key, value in (config.get("search") or {}).items()
        if key in SearchConfig.__dataclass_fields__
    })
    engine = load_engine(
        args.model,
        args.checkpoint,
        device=args.device,
        batch_size=args.batch,
        max_batch_tokens=args.max_batch_tokens,
    )
    index = SearchIndex(load_corpus(args.corpus), chunk_chars=search.chunk_chars)
    try:
        result = await search_sag2(
            index,
            LocalOpenJeVJudge(engine, args.model),
            args.query,
            **search.as_dict(),
        )
    finally:
        index.close()
    return {
        "query": args.query,
        "model": args.model,
        "top_k": search.top_k,
        "candidates": [
            {
                "rank": rank,
                "document_id": item["external_id"],
                "title": item["title"],
                "text": item["full_doc"],
                "jev_score": item["jev_score"],
            }
            for rank, item in enumerate(result["candidates"][: search.top_k], 1)
        ],
        "trace": result["trace"],
        "usage": result["usage"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--model", default="9b")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
