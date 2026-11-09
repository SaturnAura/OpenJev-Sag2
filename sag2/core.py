"""Minimal SAG2 retrieval core.

The pipeline is lexical recall with SQLite FTS5 BM25, OpenJev evidence
judgments, one or two term-expansion hops, and greedy DPP diversity.
It is intentionally independent of the model runtime so the same search
algorithm can use either OpenJev in-process or an OpenAI-compatible service.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import aiohttp

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "with",
}
WORD = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)
HAN = re.compile(r"[\u3400-\u9fff]+")
SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
HAS_CONTENT = re.compile(r"[\w]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize English and CJK text without a heavyweight segmentation model."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = WORD.findall(normalized)
    for passage in HAN.findall(normalized):
        words.extend(passage[i : i + 2] for i in range(max(1, len(passage) - 1)))
    return [
        word
        for word in words
        if word not in STOPWORDS and HAS_CONTENT.search(word)
    ]


def quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def query_expression(query: str) -> str:
    terms = list(dict.fromkeys(tokenize(query)))[:80]
    return " OR ".join(quote(term) for term in terms)


def diversity_tokens(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def chunk_text(text: str, target: int = 1500) -> list[str]:
    """Split at sentence boundaries with a small overlap between adjacent chunks."""

    text = text.strip()
    if not text:
        return []
    sentences = [part.strip() for part in SENTENCES.split(text) if part.strip()]
    if not sentences:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    limit = max(2, target - 80)
    for sentence in sentences:
        if current and size + len(sentence) + 1 > limit:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_size = 0
            for previous in reversed(current):
                if overlap and overlap_size + len(previous) + 1 > 80:
                    break
                overlap.insert(0, previous)
                overlap_size += len(previous) + 1
            current = overlap
            size = overlap_size
        current.append(sentence)
        size += len(sentence) + (1 if size else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass(frozen=True)
class SearchConfig:
    candidate_k: int = 100
    seed_k: int = 10
    max_hops: int = 2
    expansion_term_limit: int = 20
    top_k: int = 10
    chunk_chars: int = 1500

    def as_dict(self) -> dict[str, int]:
        values = asdict(self)
        values.pop("top_k")
        values.pop("chunk_chars")
        return values


class SearchIndex:
    """In-memory FTS5 index over sentence-aligned corpus chunks."""

    def __init__(self, corpus: list[dict[str, Any]], chunk_chars: int = 1500):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            PRAGMA synchronous=OFF;
            PRAGMA journal_mode=MEMORY;
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                external_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                full_doc TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunk_fts USING fts5(title, heading, body);
            """
        )
        chunk_id = 0
        documents = []
        for document_id, document in enumerate(corpus, 1):
            title = str(document.get("title") or "").strip()
            text = str(document.get("text") or "").strip()
            if not text:
                continue
            external_id = str(
                document.get("id")
                or document.get("_id")
                or document.get("source_id")
                or document_id
            )
            full_doc = f"{title}\n{text}".strip()
            source_id = str(document.get("source_id") or external_id)
            documents.append(text)
            for paragraph in chunk_text(text, chunk_chars):
                chunk_id += 1
                self.conn.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                    (
                        chunk_id,
                        document_id,
                        external_id,
                        source_id,
                        title,
                        "",
                        paragraph,
                        full_doc,
                    ),
                )
                self.conn.execute(
                    "INSERT INTO chunk_fts(rowid,title,heading,body) VALUES(?,?,?,?)",
                    (
                        chunk_id,
                        " ".join(tokenize(title)),
                        "",
                        " ".join(tokenize(paragraph)),
                    ),
                )
        self.conn.commit()
        frequencies: Counter[str] = Counter()
        for text in documents:
            frequencies.update(set(diversity_tokens(text)))
        count = len(documents)
        self.diversity_idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1
            for term, frequency in frequencies.items()
        }
        self._frequencies: dict[str, int] = {}

    def close(self) -> None:
        self.conn.close()

    def frequency(self, term: str) -> int:
        if term not in self._frequencies:
            self._frequencies[term] = self.conn.execute(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH ?",
                (f"body : {quote(term)}",),
            ).fetchone()[0]
        return self._frequencies[term]

    def retrieve(
        self,
        expression: str,
        excluded: set[int],
        limit: int,
        required_term: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["chunk_fts MATCH ?"]
        parameters: list[Any] = [expression]
        if excluded:
            markers = ",".join("?" for _ in excluded)
            clauses.append(f"chunk_fts.rowid NOT IN ({markers})")
            parameters.extend(sorted(excluded))
        if required_term is not None:
            clauses.append(
                "chunk_fts.rowid IN "
                "(SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?)"
            )
            parameters.append(quote(required_term))
        parameters.append(limit)
        sql = (
            "SELECT c.*, bm25(chunk_fts,2,2,1) AS score "
            "FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.rowid "
            f"WHERE {' AND '.join(clauses)} ORDER BY score,c.id LIMIT ?"
        )
        return [dict(row) for row in self.conn.execute(sql, parameters)]

    def expansion_terms(
        self, query: str, seeds: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        original = set(tokenize(query))
        max_score = max([0.0, *(float(seed["jev_score"]) for seed in seeds)])
        ranked: dict[str, dict[str, Any]] = {}
        total = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        for seed in seeds:
            counts: Counter[str] = Counter()
            context_scores: dict[str, float] = {}
            for sentence in SENTENCES.split(seed["content"]):
                if not sentence.strip():
                    continue
                units = tokenize(sentence)
                counts.update(units)
                context = sum(
                    math.log(
                        1
                        + (total - self.frequency(term) + 0.5)
                        / (self.frequency(term) + 0.5)
                    )
                    for term in dict.fromkeys(tokenize(sentence))
                    if term in original
                )
                for unit in units:
                    context_scores[unit] = max(context_scores.get(unit, 0.0), context)
            weight = 0.5 + 0.5 * (
                float(seed["jev_score"]) / max_score if max_score > 0 else 0.0
            )
            for term, frequency_in_seed in counts.items():
                frequency = self.frequency(term)
                if not frequency or all(word in original for word in tokenize(term)):
                    continue
                idf = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
                context = context_scores.get(term, 0.0)
                value = (
                    (1 + math.log(frequency_in_seed))
                    * idf
                    * (1 + context)
                    * weight
                )
                if term not in ranked or value > ranked[term]["score"]:
                    ranked[term] = {"term": term, "score": value}
        return sorted(ranked.values(), key=lambda item: (-item["score"], item["term"]))[
            :limit
        ]

    def retrieve_expansion(
        self,
        expression: str,
        terms: list[dict[str, Any]],
        excluded: set[int],
        limit: int,
    ) -> list[dict[str, Any]]:
        ranking = " OR ".join(quote(term["term"]) for term in terms)
        lists = [
            self.retrieve(
                ranking + f" OR ({expression})",
                excluded,
                limit,
                required_term=term["term"],
            )
            for term in terms
        ]
        hits: list[dict[str, Any]] = []
        seen: set[int] = set()
        for rank in range(limit):
            for term_index, rows in enumerate(lists):
                if rank >= len(rows) or rows[rank]["id"] in seen:
                    continue
                seen.add(rows[rank]["id"])
                hits.append(
                    {
                        **rows[rank],
                        "expansion_term": terms[term_index]["term"],
                        "expansion_term_rank": term_index + 1,
                        "expansion_bm25_rank": rank + 1,
                    }
                )
                if len(hits) == limit:
                    return hits
        return hits


class Judge(Protocol):
    async def score(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


def _score_payload(
    query: str,
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    state = json.dumps(
        {
            "original_question": query,
            "previously_retrieved_evidence": [
                {
                    "title": item["title"],
                    "passage": item["content"],
                    **(
                        {"previous_jev_score": item["previous_jev_score"]}
                        if "previous_jev_score" in item
                        else {}
                    ),
                }
                for item in evidence
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    instruction = (
        "Is this candidate itself necessary evidence in a fully supported "
        "answer chain for original_question? Use context only to connect facts. "
        "Count necessary intermediate identification facts. Do not treat topical "
        "or optional background as necessary."
    )
    questions = {
        f"candidate_{candidate['id']}": {
            "type": "bool",
            "instructions": (
                f"{instruction}\n\nCandidate title:\n{candidate['title']}\n\n"
                f"Candidate passage:\n{candidate['content']}"
            ),
            "criteria": {
                "true": "The candidate itself is necessary evidence.",
                "false": "The candidate is not itself necessary evidence.",
            },
        }
        for candidate in candidates
    }
    return state, questions


class JevClient:
    """OpenAI-compatible OpenJev client used by the optional HTTP service."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        question_type: str = "bool",
    ):
        endpoint = base_url.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint += "/decide"
        self.local = endpoint.endswith("/v1/decide")
        if not self.local and not endpoint.endswith(("/decisions", "/systemone")):
            endpoint += "/systemone"
        if question_type not in {"bool", "noul"}:
            raise ValueError("SAG2 expects a boolean judge")
        self.session = session
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.question_type = question_type

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for attempt in range(4):
            async with self.session.post(
                self.endpoint, json=payload, headers=headers
            ) as response:
                if response.status in {408, 429, 500, 502, 503, 504} and attempt < 3:
                    await response.read()
                    await asyncio.sleep(2**attempt)
                    continue
                response.raise_for_status()
                return await response.json()
        raise RuntimeError("JEV retry loop exited unexpectedly")

    def _local_payload(
        self, state: str, questions: dict[str, Any]
    ) -> dict[str, Any]:
        return {"model": self.model, "state": state, "questions": questions}

    async def score(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state, questions = _score_payload(query, candidates, evidence)
        payload = {"model": self.model, "state": state, "questions": questions}
        if self.local:
            payload = self._local_payload(state, questions)
        data = await self._request(payload)
        results = data.get("results") or {}
        scored = []
        for candidate in candidates:
            question_id = f"candidate_{candidate['id']}"
            result = results[question_id]
            probability = result.get("probabilities", {}).get("true")
            if probability is None:
                probability = result.get("probabilities", {}).get("yes")
            if probability is None:
                raise ValueError(f"OpenJev result {question_id} has no true probability")
            scored.append(
                {
                    **candidate,
                    "jev_score": float(probability),
                    "jev_model": data.get("model", self.model),
                }
            )
        usage = data.get("usage", {})
        return scored, {
            "requests": int(usage.get("requests", 1)),
            "judgments": len(candidates),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }


class LocalOpenJeVJudge:
    """In-process OpenJev judge with no HTTP serialization or server process."""

    def __init__(self, engine, model_label: str):
        self.engine = engine
        self.model_label = model_label

    async def score(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state, questions = _score_payload(query, candidates, evidence)
        response = await asyncio.to_thread(
            self.engine.decide, state, questions, share_state=True
        )
        scored = []
        for candidate in candidates:
            result = response.decisions[f"candidate_{candidate['id']}"]
            scored.append(
                {
                    **candidate,
                    "jev_score": float(result.probabilities["true"]),
                    "jev_model": self.model_label,
                }
            )
        usage = response.usage
        return scored, {
            "requests": int(usage.get("requests", 1)),
            "judgments": len(candidates),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }


def rank_dpp(candidates: list[dict[str, Any]], idf: dict[str, float]) -> list[dict[str, Any]]:
    """Greedy DPP re-ranking by evidence probability and lexical diversity."""

    vectors = []
    for candidate in candidates:
        vector = {
            term: (1 + math.log(count)) * idf.get(term, 1.0)
            for term, count in Counter(diversity_tokens(candidate["content"])).items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({term: value / norm for term, value in vector.items()})
    scores = [float(candidate["jev_score"]) for candidate in candidates]
    kernel = [
        [
            scores[left]
            * scores[right]
            * sum(
                value * vectors[right].get(term, 0.0)
                for term, value in vectors[left].items()
            )
            for right in range(len(candidates))
        ]
        for left in range(len(candidates))
    ]
    residuals = [score * score for score in scores]
    tolerance = len(candidates) * math.ulp(max(residuals))
    coefficients = [[] for _ in candidates]
    remaining = list(range(len(candidates)))
    order: list[int] = []
    while remaining:
        winner = max(remaining, key=lambda index: residuals[index])
        if residuals[winner] <= tolerance:
            order.extend(remaining)
            break
        order.append(winner)
        remaining.remove(winner)
        denominator = math.sqrt(residuals[winner]) or 1.0
        for index in remaining:
            projection = (
                kernel[winner][index]
                - sum(
                    left * right
                    for left, right in zip(
                        coefficients[winner], coefficients[index], strict=True
                    )
                )
            ) / denominator
            coefficients[index].append(projection)
            residuals[index] = max(0.0, residuals[index] - projection * projection)
    return [candidates[index] for index in order]


async def search_sag2(
    index: SearchIndex,
    client: Judge,
    query: str,
    *,
    candidate_k: int = 100,
    seed_k: int = 10,
    max_hops: int = 2,
    expansion_term_limit: int = 20,
) -> dict[str, Any]:
    if not 0 <= max_hops <= 2:
        raise ValueError("max_hops must be between 0 and 2")
    expression = query_expression(query)
    if not expression:
        raise ValueError("query produced no lexical terms")
    retained: dict[int, dict[str, Any]] = {}
    carried: dict[int, dict[str, Any]] = {}
    protected: dict[int, dict[str, Any]] = {}
    expanded: set[int] = set()
    seen: set[int] = set()
    frontier: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    usage = {"requests": 0, "judgments": 0, "input_tokens": 0, "output_tokens": 0}

    def ranked() -> list[dict[str, Any]]:
        return sorted(
            retained.values(),
            key=lambda item: (
                -item["jev_score"],
                -item["fusion_score"],
                item["id"],
            ),
        )

    for hop in range(max_hops + 1):
        if hop and not frontier:
            break
        terms = (
            index.expansion_terms(query, frontier, expansion_term_limit) if hop else []
        )
        if hop and not terms:
            break
        hits = (
            index.retrieve_expansion(expression, terms, seen, candidate_k)
            if hop
            else index.retrieve(expression, seen, candidate_k)
        )
        candidates = [
            {**hit, "hop": hop, "fusion_score": 1 / (61 + position)}
            for position, hit in enumerate(hits)
        ]
        if not candidates:
            break
        if hop == 0:
            protected = {item["id"]: item for item in candidates[:5]}
        carried.update({item["id"]: item for item in frontier})
        pool = (
            list(
                {
                    item["id"]: item
                    for item in [*protected.values(), *carried.values(), *candidates]
                }.values()
            )
            if hop
            else candidates
        )
        evidence = (
            list(
                {
                    item["id"]: item
                    for item in [*carried.values(), *candidates[:seed_k]]
                }.values()
            )
            if hop
            else []
        )
        if hop == 2:
            incoming = [item for item in pool if item["id"] not in retained][:20]
            previous = ranked()
            old = list({item["id"]: item for item in [*pool, *previous]}.values())
            pool = old[: max(0, len(pool) - len(incoming))] + incoming
            evidence = [
                {**item, "previous_jev_score": item["jev_score"]}
                for item in previous[:10]
            ]
        scored, round_usage = await client.score(query, pool, evidence)
        for key in usage:
            usage[key] += round_usage[key]
        retained = {item["id"]: item for item in scored}
        seen.update(retained)
        trace.append(
            {
                "hop": hop,
                "candidate_count": len(pool),
                "evidence_count": len(evidence),
                "terms": [term["term"] for term in terms],
            }
        )
        frontier = [
            item for item in ranked() if item["id"] not in expanded
        ][:seed_k]
        expanded.update(item["id"] for item in frontier)

    output = ranked()[:candidate_k]
    if len(output) >= 2:
        output = rank_dpp(output[:10], index.diversity_idf) + output[10:]
    return {"candidates": output, "trace": trace, "usage": usage, "implementation": "python"}

