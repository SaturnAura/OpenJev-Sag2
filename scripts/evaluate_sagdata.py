"""Evaluate sparse, dense, and OpenJev-SAG2 retrieval on D:\\SAGDataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from sag2.dense import BgeM3Retriever
from openjev.runtime import load_engine, release_engine
from sag2 import LocalOpenJeVJudge, SearchIndex, search_sag2
from sag2.core import query_expression
from sag2.metrics import (
    mean_metrics,
    percentile,
    ranking_metrics,
    unique_documents,
)
from sag2.datasets import QuerySample, load_samples

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def grouped(samples: list[QuerySample]) -> dict[str, list[QuerySample]]:
    groups: dict[str, list[QuerySample]] = {}
    for sample in samples:
        groups.setdefault(sample.dataset, []).append(sample)
    return groups


def finish_dataset(
    dataset: str,
    rows: list[dict[str, Any]],
    system: str,
) -> dict[str, Any]:
    query_times = [row["query_seconds"] for row in rows]
    metrics = mean_metrics(rows, system)
    return {
        "dataset": dataset,
        "queries": len(rows),
        "ndcg@10": metrics["ndcg@10"],
        "recall@2": metrics["recall@2"],
        "recall@10": metrics["recall@10"],
        "mrr@10": metrics["mrr@10"],
        "index_seconds": sum(row["index_seconds"] for row in rows),
        "query_seconds": sum(query_times),
        "query_ms_p50": 1000 * percentile(query_times, 0.50),
        "query_ms_p95": 1000 * percentile(query_times, 0.95),
        "queries_detail": rows,
    }


def evaluate_bm25(samples: list[QuerySample]) -> dict[str, Any]:
    payload: dict[str, Any] = {"system": "bm25", "datasets": {}}
    for dataset, group in grouped(samples).items():
        rows = []
        for sample in group:
            started = time.perf_counter()
            index = SearchIndex(list(sample.documents), chunk_chars=1000)
            index_seconds = time.perf_counter() - started
            try:
                started = time.perf_counter()
                hits = index.retrieve(
                    query_expression(sample.question), set(), len(sample.documents)
                )
                query_seconds = time.perf_counter() - started
            finally:
                index.close()
            rows.append(
                {
                    "query_id": sample.query_id,
                    "question": sample.question,
                    "relevant": len(sample.relevant),
                    "candidate_documents": len(sample.documents),
                    "bm25": ranking_metrics(
                        unique_documents(hits), sample.relevant
                    ),
                    "index_seconds": index_seconds,
                    "query_seconds": query_seconds,
                }
            )
        payload["datasets"][dataset] = finish_dataset(dataset, rows, "bm25")
        print(f"[BM25] {dataset}: {len(rows)} queries", flush=True)
    return payload


def evaluate_bgem3(samples: list[QuerySample], args) -> dict[str, Any]:
    retriever = BgeM3Retriever(
        args.model_path,
        device=args.device,
        batch_size=args.batch,
        max_length=args.max_length,
    )
    payload: dict[str, Any] = {
        "system": "bge-m3",
        "model_path": str(Path(args.model_path).resolve()),
        "load_seconds": retriever.load_seconds,
        "dtype": str(retriever.dtype),
        "batch_size": args.batch,
        "max_length": args.max_length,
        "datasets": {},
    }
    for dataset, group in grouped(samples).items():
        rows = []
        for sample in group:
            started = time.perf_counter()
            retriever.build(list(sample.documents))
            index_seconds = time.perf_counter() - started
            ranked, _scores, query_seconds = retriever.search(
                sample.question,
                top_k=args.top_k,
                document_ids=[document["id"] for document in sample.documents],
            )
            rows.append(
                {
                    "query_id": sample.query_id,
                    "question": sample.question,
                    "relevant": len(sample.relevant),
                    "candidate_documents": len(sample.documents),
                    "bge-m3": ranking_metrics(ranked, sample.relevant),
                    "retrieved_ids": ranked,
                    "index_seconds": index_seconds,
                    "query_seconds": query_seconds,
                }
            )
            retriever.clear()
        payload["datasets"][dataset] = finish_dataset(dataset, rows, "bge-m3")
        print(f"[BGE-M3] {dataset}: {len(rows)} queries", flush=True)
    retriever.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


async def evaluate_openjev(
    samples: list[QuerySample], model: str, args
) -> dict[str, Any]:
    load_started = time.perf_counter()
    engine = load_engine(
        model,
        args.checkpoint,
        device=args.device,
        batch_size=args.batch,
        max_batch_tokens=args.max_batch_tokens,
    )
    load_seconds = time.perf_counter() - load_started
    system = model.casefold()
    judge = LocalOpenJeVJudge(engine, system)
    payload: dict[str, Any] = {
        "system": system,
        "model": engine.config.model_path,
        "checkpoint": engine.config.readout_checkpoint,
        "load_seconds": load_seconds,
        "datasets": {},
    }
    try:
        for dataset, group in grouped(samples).items():
            rows = []
            for sample in group:
                started = time.perf_counter()
                index = SearchIndex(list(sample.documents), chunk_chars=1000)
                index_seconds = time.perf_counter() - started
                try:
                    started = time.perf_counter()
                    result = await search_sag2(
                        index,
                        judge,
                        sample.question,
                        candidate_k=min(args.candidate_k, len(sample.documents)),
                        seed_k=min(args.seed_k, len(sample.documents)),
                        max_hops=args.max_hops,
                        expansion_term_limit=args.expansion_terms,
                    )
                    query_seconds = time.perf_counter() - started
                finally:
                    index.close()
                    engine.clear_state_cache()
                ranked = unique_documents(result["candidates"])
                rows.append(
                    {
                        "query_id": sample.query_id,
                        "question": sample.question,
                        "relevant": len(sample.relevant),
                        "candidate_documents": len(sample.documents),
                        system: ranking_metrics(ranked, sample.relevant),
                        "retrieved_ids": ranked,
                        "index_seconds": index_seconds,
                        "query_seconds": query_seconds,
                        "usage": result["usage"],
                    }
                )
            payload["datasets"][dataset] = finish_dataset(dataset, rows, system)
            print(f"[{system.upper()}] {dataset}: {len(rows)} queries", flush=True)
    finally:
        release_engine(engine)
    return payload


def write_report() -> None:
    files = {
        label: ROOT / "results" / f"sagdataset_{label}.json"
        for label in ("bm25", "bge-m3", "9b")
    }
    payloads = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in files.items()
        if path.is_file()
    }
    labels = [label for label in ("bm25", "bge-m3", "9b") if label in payloads]
    display = {
        "bm25": "BM25",
        "bge-m3": "BGE-M3 Dense",
        "9b": "OpenJev-9B + SAG2",
    }
    datasets = ("2WikiMultiHopQA", "HotpotQA", "MuSiQue")
    lines = [
        "# SAGDataset 多跳检索实验结果",
        "",
        "> 从每个问题的候选段落中检索 supporting facts。"
        "指标为文档级 `nDCG@10` 与 `Recall@2`。",
        "",
        "| 数据集 | 系统 | 查询数 | nDCG@10 | Recall@2 | Recall@10 | MRR@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for label in labels:
            row = payloads[label]["datasets"].get(dataset)
            if not row:
                continue
            lines.append(
                f"| {dataset} | {display[label]} | {row['queries']} | "
                f"{row['ndcg@10']:.4f} | {row['recall@2']:.4f} | "
                f"{row['recall@10']:.4f} | {row['mrr@10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 数据集宏平均",
            "",
            "| 系统 | nDCG@10 | Recall@2 | Recall@10 | MRR@10 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in labels:
        rows = list(payloads[label]["datasets"].values())
        lines.append(
            f"| {display[label]} | "
            f"{sum(row['ndcg@10'] for row in rows) / len(rows):.4f} | "
            f"{sum(row['recall@2'] for row in rows) / len(rows):.4f} | "
            f"{sum(row['recall@10'] for row in rows) / len(rows):.4f} | "
            f"{sum(row['mrr@10'] for row in rows) / len(rows):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 运行开销",
            "",
            "| 系统 | 模型加载 |",
            "| --- | ---: |",
        ]
    )
    for label in labels:
        load_seconds = payloads[label].get("load_seconds")
        lines.append(
            f"| {display[label]} | "
            f"{f'{load_seconds:.1f} s' if load_seconds else '不适用'} |"
        )
    lines.extend(
        [
            "",
            "| 数据集 | 系统 | 上下文索引 p50 | 查询均值 | 查询 p50 | 查询 p95 | "
            "总查询时间 | QPS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in datasets:
        for label in labels:
            row = payloads[label]["datasets"].get(dataset)
            if not row:
                continue
            details = row["queries_detail"]
            index_p50 = 1000 * percentile(
                [item["index_seconds"] for item in details], 0.50
            )
            query_seconds = row["query_seconds"]
            query_mean = 1000 * query_seconds / row["queries"]
            qps = row["queries"] / query_seconds if query_seconds else 0.0
            lines.append(
                f"| {dataset} | {display[label]} | {index_p50:.1f} ms | "
                f"{query_mean:.1f} ms | {row['query_ms_p50']:.1f} ms | "
                f"{row['query_ms_p95']:.1f} ms | {query_seconds:.2f} s | "
                f"{qps:.2f} |"
            )
    if "9b" in payloads:
        lines.extend(
            [
                "",
                "## 9B 判断与 Token 用量",
                "",
                "| 数据集 | 请求数 | 判断数 | 输入 Token | 输出 Token |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for dataset, row in payloads["9b"]["datasets"].items():
            usages = [item["usage"] for item in row["queries_detail"]]
            lines.append(
                f"| {dataset} | {sum(item['requests'] for item in usages)} | "
                f"{sum(item['judgments'] for item in usages)} | "
                f"{sum(item['input_tokens'] for item in usages):,} | "
                f"{sum(item['output_tokens'] for item in usages):,} |"
            )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- 2WikiMultiHopQA 使用 `dev.parquet`；MuSiQue 使用 "
            "`musique_ans_v1.0_dev.jsonl`；HotpotQA 使用 validation。",
            "- 每个问题的候选段落即该条数据提供的 context。"
            "supporting title 作为金标文档。",
            "- HotpotQA 仅保留 supporting title 确实存在于同条 context 的样本；"
            "当前 parquet 中可用约 28%。",
            "- 每套数据固定抽样 20 个问题，seed=0；所有系统使用完全相同的样本。",
            "- BGE-M3 与 BM25 的上下文索引和查询时间分开报告；"
            "SAG2 在线查询包含 BM25、必要时两轮扩展、OpenJev 判断和 DPP。",
            "- 9B 使用完整 GPU 推理，没有 CPU layer offload。",
            "- 9B 仍是 zero-shot retrieval：动态问题 ID 未命中混训头的任务槽，"
            "`slot_head` 按设计回退到保留 A/B 行。",
            "",
        ]
    )
    (ROOT / "results" / "SAGDATASET_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


async def async_main(args) -> int:
    samples, eligible = load_samples(
        args.datasets,
        count=args.queries,
        seed=args.seed,
        data_root=args.data_root,
    )
    manifest = {
        "source": str(Path(args.data_root).resolve()),
        "seed": args.seed,
        "requested_per_dataset": args.queries,
        "eligible_counts": eligible,
        "selected": {
            dataset: [sample.query_id for sample in grouped(samples)[dataset]]
            for dataset in args.datasets
        },
    }
    save(ROOT / "results" / "sagdataset_manifest.json", manifest)

    systems = [value.strip() for value in args.systems.split(",") if value.strip()]
    for system in systems:
        if system == "bm25":
            payload = evaluate_bm25(samples)
            output_name = "bm25"
        elif system == "bgem3":
            payload = evaluate_bgem3(samples, args)
            output_name = "bge-m3"
        elif system == "9b":
            payload = await evaluate_openjev(samples, system, args)
            output_name = system
        else:
            raise ValueError(f"unknown system: {system}")
        save(ROOT / "results" / f"sagdataset_{output_name}.json", payload)
        write_report()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--datasets", default="2WikiMultiHopQA,HotpotQA,MuSiQue"
    )
    result.add_argument("--queries", type=int, default=20)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--systems", default="bm25,bgem3,9b")
    result.add_argument("--data-root", default=r"D:\SAGDataset")
    result.add_argument("--candidate-k", type=int, default=20)
    result.add_argument("--seed-k", type=int, default=5)
    result.add_argument("--max-hops", type=int, default=2)
    result.add_argument("--expansion-terms", type=int, default=8)
    result.add_argument("--top-k", type=int, default=10)
    result.add_argument("--checkpoint", default=None)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch", type=int, default=8)
    result.add_argument("--max-batch-tokens", type=int, default=4096)
    result.add_argument("--model-path", default=r"C:\bge-m3")
    result.add_argument("--max-length", type=int, default=512)
    result.add_argument("--report-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    args.datasets = [
        value.strip() for value in args.datasets.split(",") if value.strip()
    ]
    if args.report_only:
        write_report()
        print(f"Wrote {ROOT / 'results' / 'SAGDATASET_RESULTS.md'}")
        return 0
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
