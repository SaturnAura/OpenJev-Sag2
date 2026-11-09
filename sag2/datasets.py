"""Load 2WikiMultiHopQA, HotpotQA, and MuSiQue as multi-hop retrieval tasks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

@dataclass(frozen=True)
class QuerySample:
    dataset: str
    query_id: str
    question: str
    documents: tuple[dict[str, str], ...]
    relevant: dict[str, int]


def _merge_documents(
    rows: list[tuple[str, str]], relevant_titles: set[str]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    merged: dict[str, list[str]] = {}
    for title, text in rows:
        if title and text:
            merged.setdefault(title, []).append(text)
    documents = [
        {"id": title, "title": title, "text": " ".join(parts)}
        for title, parts in merged.items()
    ]
    available = {document["id"] for document in documents}
    if not relevant_titles or not relevant_titles <= available:
        return [], {}
    return documents, {title: 1 for title in sorted(relevant_titles)}


def _load_2wiki(path: Path) -> list[QuerySample]:
    samples = []
    for row in pq.read_table(path).to_pylist():
        context = json.loads(row["context"])
        facts = json.loads(row["supporting_facts"])
        supporting = {str(title) for title, _sentence_id in facts}
        documents, relevant = _merge_documents(
            [(str(title), " ".join(map(str, sentences))) for title, sentences in context],
            supporting,
        )
        if documents:
            samples.append(
                QuerySample(
                    "2WikiMultiHopQA",
                    str(row["_id"]),
                    str(row["question"]),
                    tuple(documents),
                    relevant,
                )
            )
    return samples


def _load_hotpot(path: Path) -> list[QuerySample]:
    samples = []
    for row in pq.read_table(path).to_pylist():
        context = row["context"]
        facts = row["supporting_facts"]
        supporting = {str(title) for title in facts["title"]}
        documents, relevant = _merge_documents(
            [
                (str(title), " ".join(map(str, sentences)))
                for title, sentences in zip(
                    context["title"], context["sentences"], strict=True
                )
            ],
            supporting,
        )
        if documents:
            samples.append(
                QuerySample(
                    "HotpotQA",
                    str(row["id"]),
                    str(row["question"]),
                    tuple(documents),
                    relevant,
                )
            )
    return samples


def _load_musique(path: Path) -> list[QuerySample]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("answerable") is False:
                continue
            supporting = {
                str(paragraph["title"])
                for paragraph in row["paragraphs"]
                if paragraph.get("is_supporting")
            }
            documents, relevant = _merge_documents(
                [
                    (str(paragraph["title"]), str(paragraph["paragraph_text"]))
                    for paragraph in row["paragraphs"]
                ],
                supporting,
            )
            if documents:
                samples.append(
                    QuerySample(
                        "MuSiQue",
                        str(row["id"]),
                        str(row["question"]),
                        tuple(documents),
                        relevant,
                    )
                )
    return samples


LOADERS = {
    "2WikiMultiHopQA": (
        Path(r"D:\SAGDataset\2WikiMultiHopQA\dev.parquet"),
        _load_2wiki,
    ),
    "HotpotQA": (
        Path(r"D:\SAGDataset\HotpotQA\validation-00000-of-00001.parquet"),
        _load_hotpot,
    ),
    "MuSiQue": (
        Path(r"D:\SAGDataset\musique\musique_ans_v1.0_dev.jsonl"),
        _load_musique,
    ),
}


def load_samples(
    datasets: list[str], *, count: int, seed: int, data_root: str | Path | None = None
) -> tuple[list[QuerySample], dict[str, int]]:
    selected = []
    eligible_counts = {}
    for dataset in datasets:
        path, loader = LOADERS[dataset]
        if data_root:
            path = Path(data_root) / path.relative_to(r"D:\SAGDataset")
        rows = loader(path)
        eligible_counts[dataset] = len(rows)
        if count > 0:
            rows = random.Random(seed).sample(rows, min(count, len(rows)))
        selected.extend(sorted(rows, key=lambda row: row.query_id))
    return selected, eligible_counts

