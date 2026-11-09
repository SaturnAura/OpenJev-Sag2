"""Minimal in-memory BGE-M3 dense retriever."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BgeM3Retriever:
    """Encode documents with BGE-M3 CLS pooling and retrieve by cosine similarity."""

    def __init__(
        self,
        model_path: str | Path = r"C:\bge-m3",
        *,
        device: str = "cuda:0",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - started
        self.document_embeddings: torch.Tensor | None = None

    @torch.inference_mode()
    def encode(self, texts: list[str], *, show_progress: bool = False) -> torch.Tensor:
        """Return normalized 1024-dimensional BGE-M3 CLS embeddings."""

        batches = []
        started = time.perf_counter()
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, total=(len(texts) + self.batch_size - 1) // self.batch_size)
            except ImportError:
                pass
        for start in iterator:
            tokens = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            output = self.model(**tokens, return_dict=True)
            cls = output.last_hidden_state[:, 0].float()
            batches.append(F.normalize(cls, p=2, dim=-1).cpu())
        self.encode_seconds = time.perf_counter() - started
        return torch.cat(batches, dim=0) if batches else torch.empty((0, 1024))

    def build(
        self, documents: list[dict], *, show_progress: bool = False
    ) -> dict[str, float | int]:
        """Encode corpus documents and move the compact index onto the device."""

        texts = [
            f"{str(document.get('title') or '').strip()}\n"
            f"{str(document.get('text') or '').strip()}".strip()
            for document in documents
        ]
        started = time.perf_counter()
        embeddings = self.encode(texts, show_progress=show_progress)
        self.document_embeddings = embeddings.to(self.device)
        elapsed = time.perf_counter() - started
        return {
            "documents": len(documents),
            "embedding_dimension": int(embeddings.shape[1]),
            "seconds": elapsed,
            "documents_per_second": len(documents) / elapsed if elapsed else 0.0,
        }

    @torch.inference_mode()
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> tuple[list[str], list[float], float]:
        """Return document ids, cosine scores, and query latency in seconds."""

        if self.document_embeddings is None:
            raise RuntimeError("build() must be called before search()")
        started = time.perf_counter()
        query_embedding = self.encode([query]).to(self.device)
        scores = self.document_embeddings @ query_embedding[0]
        count = min(top_k, scores.shape[0])
        values, indices = torch.topk(scores, count)
        ids = document_ids or list(range(scores.shape[0]))
        ranked = [str(ids[int(index)]) for index in indices.cpu()]
        elapsed = time.perf_counter() - started
        return ranked, values.float().cpu().tolist(), elapsed

    def clear(self) -> None:
        self.document_embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

