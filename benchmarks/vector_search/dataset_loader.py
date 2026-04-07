from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "data" / "public_mini_eval.json"
)
DEFAULT_PUBLIC_CORPUS_CACHE = (
    Path(__file__).resolve().parents[2] / "artifacts" / "vector-search-corpora"
)
PUBLIC_CORPUS_SPECS: dict[str, dict[str, str]] = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "description": (
            "Opt-in BEIR SciFact retrieval corpus. This repo does not redistribute "
            "the third-party dataset; review upstream terms before use."
        ),
        "license": "Third-party public corpus; review upstream SciFact and BEIR terms before use.",
    }
}


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


def _download_public_corpus(dataset_name: str, *, cache_dir: Path) -> Path:
    if dataset_name not in PUBLIC_CORPUS_SPECS:
        raise ValueError(f"Unsupported public corpus: {dataset_name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = PUBLIC_CORPUS_SPECS[dataset_name]
    zip_path = cache_dir / f"{dataset_name}.zip"
    with urllib.request.urlopen(spec["url"]) as response, zip_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_dir)
    return zip_path


def _find_beir_root(cache_dir: Path) -> Path:
    for corpus_path in cache_dir.rglob("corpus.jsonl"):
        dataset_root = corpus_path.parent
        qrels_path = dataset_root / "qrels" / "test.tsv"
        queries_path = dataset_root / "queries.jsonl"
        if qrels_path.is_file() and queries_path.is_file():
            return dataset_root
    raise FileNotFoundError(
        f"No BEIR-style dataset root found under {cache_dir}. "
        "Expected corpus.jsonl, queries.jsonl, and qrels/test.tsv."
    )


def _ensure_public_corpus(
    dataset_name: str,
    *,
    cache_dir: Path,
    allow_download: bool,
) -> Path:
    dataset_root = cache_dir / dataset_name
    expected_paths = (
        dataset_root / "corpus.jsonl",
        dataset_root / "queries.jsonl",
        dataset_root / "qrels" / "test.tsv",
    )
    if all(path.is_file() for path in expected_paths):
        return dataset_root

    if not allow_download:
        raise FileNotFoundError(
            f"Public corpus '{dataset_name}' is not cached under {cache_dir}. "
            "Run again with --download-public-corpus to fetch it explicitly."
        )

    _download_public_corpus(dataset_name, cache_dir=cache_dir)
    return _find_beir_root(cache_dir)


def load_public_corpus(
    dataset_name: str,
    *,
    cache_dir: str | Path = DEFAULT_PUBLIC_CORPUS_CACHE,
    allow_download: bool = False,
    split: str = "test",
    max_docs: int | None = None,
    max_queries: int | None = None,
) -> VectorSearchDataset:
    if dataset_name not in PUBLIC_CORPUS_SPECS:
        raise ValueError(
            f"Unsupported public corpus '{dataset_name}'. "
            f"Available: {sorted(PUBLIC_CORPUS_SPECS)}"
        )

    cache_path = Path(cache_dir)
    dataset_root = _ensure_public_corpus(
        dataset_name,
        cache_dir=cache_path,
        allow_download=allow_download,
    )

    corpus: dict[str, str] = {}
    with (dataset_root / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            title = str(record.get("title", "")).strip()
            text = str(record.get("text", "")).strip()
            doc_text = " ".join(part for part in (title, text) if part)
            corpus[str(record["_id"])] = doc_text

    queries: dict[str, str] = {}
    with (dataset_root / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            queries[str(record["_id"])] = str(record["text"])

    qrels_path = dataset_root / "qrels" / f"{split}.tsv"
    qrels: dict[str, set[str]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row["score"])
            if score > 0 and query_id in queries and doc_id in corpus:
                qrels.setdefault(query_id, set()).add(doc_id)

    selected_query_ids = sorted(qrels.keys())
    if max_queries is not None:
        selected_query_ids = selected_query_ids[:max_queries]

    relevant_doc_ids = {
        doc_id for query_id in selected_query_ids for doc_id in qrels[query_id]
    }
    all_doc_ids = sorted(corpus.keys())

    if max_docs is not None:
        if len(relevant_doc_ids) > max_docs:
            raise ValueError(
                f"max_docs={max_docs} is smaller than the number of relevant docs "
                f"required by the selected queries ({len(relevant_doc_ids)})."
            )
        selected_doc_ids = sorted(relevant_doc_ids)
        remaining = [doc_id for doc_id in all_doc_ids if doc_id not in relevant_doc_ids]
        selected_doc_ids.extend(remaining[: max_docs - len(selected_doc_ids)])
    else:
        selected_doc_ids = all_doc_ids

    selected_doc_id_set = set(selected_doc_ids)
    query_records: list[VectorSearchQuery] = []
    for query_id in selected_query_ids:
        filtered_relevant = tuple(
            doc_id for doc_id in sorted(qrels[query_id]) if doc_id in selected_doc_id_set
        )
        if not filtered_relevant:
            continue
        query_records.append(
            VectorSearchQuery(
                query_id=query_id,
                text=queries[query_id],
                relevant_ids=filtered_relevant,
            )
        )

    spec = PUBLIC_CORPUS_SPECS[dataset_name]
    return VectorSearchDataset(
        dataset_name=f"beir_{dataset_name}_{split}",
        license=spec["license"],
        description=(
            f"{spec['description']} Selected {len(selected_doc_ids)} docs and "
            f"{len(query_records)} queries from the {split} split."
        ),
        doc_ids=tuple(selected_doc_ids),
        doc_texts=tuple(corpus[doc_id] for doc_id in selected_doc_ids),
        queries=tuple(query_records),
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
