from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "data" / "public_mini_eval.json"
)


@dataclass(slots=True)
class VectorSearchQuery:
    query_id: str
    text: str
    relevant_ids: tuple[str, ...]


@dataclass(slots=True)
class VectorSearchDataset:
    dataset_name: str
    license: str
    description: str
    doc_ids: tuple[str, ...]
    doc_texts: tuple[str, ...]
    queries: tuple[VectorSearchQuery, ...]


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> VectorSearchDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return VectorSearchDataset(
        dataset_name=str(payload["dataset_name"]),
        license=str(payload["license"]),
        description=str(payload["description"]),
        doc_ids=tuple(str(item["id"]) for item in payload["docs"]),
        doc_texts=tuple(str(item["text"]) for item in payload["docs"]),
        queries=tuple(
            VectorSearchQuery(
                query_id=str(item["id"]),
                text=str(item["text"]),
                relevant_ids=tuple(str(doc_id) for doc_id in item["relevant_ids"]),
            )
            for item in payload["queries"]
        ),
    )


def _hashed_indices(token: str, dim: int) -> tuple[tuple[int, float], tuple[int, float]]:
    outputs: list[tuple[int, float]] = []
    for salt in ("a", "b"):
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        outputs.append((index, sign))
    return outputs[0], outputs[1]


def embed_text(text: str, *, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = TOKEN_RE.findall(text.lower())
    for token in tokens:
        for index, sign in _hashed_indices(token, dim):
            vec[index] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def embed_dataset(dataset: VectorSearchDataset, *, dim: int = 128) -> tuple[np.ndarray, np.ndarray]:
    doc_matrix = np.stack([embed_text(text, dim=dim) for text in dataset.doc_texts])
    query_matrix = np.stack([embed_text(query.text, dim=dim) for query in dataset.queries])
    return doc_matrix.astype(np.float32), query_matrix.astype(np.float32)
