#!/usr/bin/env python3
# flake8: noqa: E402
"""Run a small research-only vector-search evaluation.

This lane is intentionally outside the supported product contract. It provides a
bundled, reproducible retrieval benchmark so the repo can start measuring the
paper's broader retrieval framing without pretending that vector search is a
supported runtime surface.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlx.core as mx
import numpy as np

from benchmarks.runtime_cert.utils import collect_environment_metadata, ensure_artifact_dir, write_json
from benchmarks.vector_search.dataset_loader import DEFAULT_DATASET_PATH, embed_dataset, load_dataset
from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.runtime.attention import score_block

SUMMARY_JSON = "vector_search_summary.json"
METRICS_CSV = "vector_search_metrics.csv"
SUMMARY_MD = "vector_search_summary.md"
DEFAULT_PRESETS = ("paper_mse", "paper_prod_qjl", "polarquant_exp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bundled research-only vector-search evaluation."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the vector-search artifacts.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the bundled or user-supplied vector-search dataset JSON.",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(DEFAULT_PRESETS),
        help="Preset names to compare against the dense baseline.",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3])
    return parser.parse_args()


def _dense_search(doc_embeddings: mx.array, query_embeddings: mx.array) -> mx.array:
    scores = query_embeddings @ mx.swapaxes(doc_embeddings, -1, -2)
    mx.eval(scores)
    return scores


def _approx_search(
    config: TurboQuantConfig,
    doc_embeddings: mx.array,
    query_embeddings: mx.array,
) -> tuple[mx.array, int, float, float]:
    pipe = TurboQuantPipeline(config)
    t0 = time.perf_counter()
    block = pipe.encode_k(doc_embeddings)
    mx.eval(block.packed_main if block.packed_main is not None else mx.array(0.0))
    index_build_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scores = score_block(
        query_embeddings,
        block,
        config=config,
        dequantize_main=pipe._get_k_quant().dequantize,
    )
    mx.eval(scores)
    query_ms = (time.perf_counter() - t0) * 1000.0
    return scores, block.byte_size(), index_build_ms, query_ms


def _topk_indices(scores: mx.array, k: int) -> list[list[int]]:
    score_array = np.asarray(scores)
    topk = np.argsort(-score_array, axis=-1)[..., :k]
    return [[int(item) for item in row] for row in topk[0]]


def _recall_at_k(dataset, topk_indices: list[list[int]], k: int) -> float:
    hits = 0
    for query, predicted in zip(dataset.queries, topk_indices):
        predicted_ids = {dataset.doc_ids[index] for index in predicted[:k]}
        if predicted_ids.intersection(query.relevant_ids):
            hits += 1
    return hits / len(dataset.queries)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "classification",
        "index_bytes",
        "compression_ratio",
        "index_build_ms",
        "query_ms_total",
        "query_ms_per_query",
        "recall_at_1",
        "recall_at_3",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Vector Search Summary",
        "",
        "This is a research-only vector-search evaluation on the bundled mini public dataset.",
        "It does not make vector search part of the supported product contract.",
        "",
        "## Dataset",
        "",
        f"- dataset: `{payload['dataset']['dataset_name']}`",
        f"- license: `{payload['dataset']['license']}`",
        f"- docs: `{payload['dataset']['doc_count']}`",
        f"- queries: `{payload['dataset']['query_count']}`",
        f"- embedding dim: `{payload['dataset']['embedding_dim']}`",
        "",
        "## Results",
        "",
        "| Method | Classification | Recall@1 | Recall@3 | Compression ratio | Index build ms | Query ms/query |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["evaluations"]:
        lines.append(
            "| "
            f"`{row['method']}` | {row['classification']} | "
            f"{float(row['recall_at_1']):.3f} | {float(row['recall_at_3']):.3f} | "
            f"{float(row['compression_ratio']):.2f}x | {float(row['index_build_ms']):.2f} | "
            f"{float(row['query_ms_per_query']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Honest Takeaways",
            "",
            "- This bundled mini dataset is meant to make vector-search measurement reproducible, not authoritative.",
            "- The dense baseline defines the uncompressed retrieval reference on this dataset.",
            "- Compressed rows show how recall, memory, index-build time, and query behavior move under the current preset surfaces.",
            "- These results are research-only and should not be read as supported runtime claims.",
            "",
            "## Companion Artifacts",
            "",
            f"- `{SUMMARY_JSON}` — structured research summary",
            f"- `{METRICS_CSV}` — row-oriented metric table",
            f"- `{SUMMARY_MD}` — human-readable summary",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    output_dir = ensure_artifact_dir(args.output_dir)
    dataset = load_dataset(args.dataset)
    doc_np, query_np = embed_dataset(dataset, dim=args.embedding_dim)
    doc_embeddings = mx.array(doc_np[None, ...], dtype=mx.float32)
    query_embeddings = mx.array(query_np[None, ...], dtype=mx.float32)

    dense_scores = _dense_search(doc_embeddings, query_embeddings)
    dense_index_bytes = int(doc_embeddings.nbytes)
    dense_top1 = _topk_indices(dense_scores, 1)
    dense_top3 = _topk_indices(dense_scores, 3)

    rows: list[dict[str, Any]] = [
        {
            "method": "dense",
            "classification": "reference",
            "index_bytes": dense_index_bytes,
            "compression_ratio": 1.0,
            "index_build_ms": 0.0,
            "query_ms_total": 0.0,
            "query_ms_per_query": 0.0,
            "recall_at_1": _recall_at_k(dataset, dense_top1, 1),
            "recall_at_3": _recall_at_k(dataset, dense_top3, 3),
        }
    ]

    for preset in args.presets:
        config = TurboQuantConfig.from_preset(preset)
        scores, index_bytes, build_ms, query_ms = _approx_search(
            config,
            doc_embeddings,
            query_embeddings,
        )
        top1 = _topk_indices(scores, 1)
        top3 = _topk_indices(scores, 3)
        row = {
            "method": preset,
            "classification": TurboQuantConfig.preset_metadata(preset)["classification"],
            "index_bytes": index_bytes,
            "compression_ratio": (
                dense_index_bytes / index_bytes if index_bytes > 0 else float("inf")
            ),
            "index_build_ms": build_ms,
            "query_ms_total": query_ms,
            "query_ms_per_query": query_ms / len(dataset.queries),
            "recall_at_1": _recall_at_k(dataset, top1, 1),
            "recall_at_3": _recall_at_k(dataset, top3, 3),
        }
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise SystemExit(f"non-finite vector-search metric for {preset}: {row}")
        rows.append(row)

    payload = {
        "schema_version": "1",
        "metric_family": "vector_search",
        "support_scope": "research-only",
        "status": "ok",
        "environment": collect_environment_metadata(
            model="vector-search-mini",
            mode="vector_search",
        ),
        "dataset": {
            "dataset_name": dataset.dataset_name,
            "license": dataset.license,
            "description": dataset.description,
            "path": str(Path(args.dataset).resolve()),
            "doc_count": len(dataset.doc_ids),
            "query_count": len(dataset.queries),
            "embedding_dim": args.embedding_dim,
            "top_k": list(args.top_k),
        },
        "evaluations": rows,
        "companion_artifacts": [SUMMARY_JSON, METRICS_CSV, SUMMARY_MD],
        "notes": [
            "This bundled mini dataset is a reproducible research entry point, not a production retrieval benchmark.",
            "Recall is measured against bundled relevance labels, not against a larger public leaderboard.",
            "Vector-search results remain explicitly outside the supported Apple-MLX product contract.",
        ],
    }

    summary_path = output_dir / SUMMARY_JSON
    csv_path = output_dir / METRICS_CSV
    markdown_path = output_dir / SUMMARY_MD
    write_json(summary_path, payload)
    _write_metrics_csv(csv_path, rows)
    markdown_path.write_text(_render_markdown_summary(payload), encoding="utf-8")

    print(summary_path)
    print(csv_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
