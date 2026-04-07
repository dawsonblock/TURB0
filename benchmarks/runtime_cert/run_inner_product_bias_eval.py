#!/usr/bin/env python3
# flake8: noqa: E402
"""Emit a research-only inner-product bias summary into a runtime-cert artifact.

This script measures the current paper-facing score-estimation behavior on a
fixed synthetic workload. It is intentionally research-scoped:

- `paper_mse` is the scalar-only baseline.
- `paper_prod_qjl` is the scalar main stage plus QJL residual score estimate.

The output is a provenance-backed JSON summary that can be retained alongside a
runtime-cert artifact and rendered into a dated benchmark snapshot. It is not a
release gate by itself.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlx.core as mx

from benchmarks.runtime_cert.utils import collect_environment_metadata, ensure_artifact_dir, write_json
from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.core.rotation import FixedRotation
from turboquant.runtime.attention import score_block

SUMMARY_JSON = "inner_product_bias_summary.json"
METRICS_CSV = "inner_product_bias_metrics.csv"
SUMMARY_MD = "inner_product_bias_summary.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure paper-facing inner-product bias on a synthetic workload."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive inner_product_bias_summary.json",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=12)
    parser.add_argument("--k-len", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=128)
    parser.add_argument("--q-seed", type=int, default=0)
    parser.add_argument("--k-seed", type=int, default=1)
    return parser.parse_args()


def _synthetic_queries_and_keys(args: argparse.Namespace) -> tuple[mx.array, mx.array]:
    queries = mx.random.normal(
        shape=(args.batch, args.q_len, args.d_head),
        key=mx.random.key(args.q_seed),
    )
    keys = mx.random.normal(
        shape=(args.batch, args.k_len, args.d_head),
        key=mx.random.key(args.k_seed),
    )
    return queries, keys


def _true_scores(config: TurboQuantConfig, q: mx.array, k: mx.array) -> mx.array:
    rotation = FixedRotation.from_config(config, int(q.shape[-1]))
    q_rot = rotation.apply(q.astype(mx.float32))
    k_rot = rotation.apply(k.astype(mx.float32))
    return q_rot @ mx.swapaxes(k_rot, -1, -2)


def _score_stats(config: TurboQuantConfig, q: mx.array, k: mx.array) -> dict[str, float | str]:
    pipe = TurboQuantPipeline(config)
    block = pipe.encode_k(k)
    estimated = score_block(
        q,
        block,
        config=config,
        dequantize_main=pipe._get_k_quant().dequantize,
    )
    true = _true_scores(config, q, k)

    error = estimated - true
    flat_error = error.reshape(-1)
    flat_abs_error = mx.abs(error).reshape(-1)
    mean_abs_true = float(mx.mean(mx.abs(true)).item())
    q95_index = max(int(flat_abs_error.shape[0] * 0.95) - 1, 0)
    q95_abs_error = float(mx.sort(flat_abs_error)[q95_index].item())

    return {
        "algorithm": config.algorithm_family(),
        "residual_mode": config.residual_mode,
        "quantizer_mode": config.quantizer_mode,
        "mean_signed_error": float(mx.mean(flat_error).item()),
        "mean_abs_error": float(mx.mean(flat_abs_error).item()),
        "error_variance": float(mx.var(flat_error).item()),
        "q95_abs_error": q95_abs_error,
        "mean_abs_true": mean_abs_true,
        "normalized_mean_bias": float(mx.mean(flat_error).item()) / mean_abs_true,
        "normalized_mean_abs_error": float(mx.mean(flat_abs_error).item())
        / mean_abs_true,
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "algorithm",
        "residual_mode",
        "quantizer_mode",
        "mean_signed_error",
        "mean_abs_error",
        "error_variance",
        "q95_abs_error",
        "mean_abs_true",
        "normalized_mean_bias",
        "normalized_mean_abs_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_summary(payload: dict[str, Any]) -> str:
    workload = payload["workload"]
    algorithms = payload["algorithms"]
    comparison = payload["comparison"]

    lines = [
        "# Inner-Product Bias Summary",
        "",
        "This is a research-only score diagnostic for the paper-facing scalar-only and two-stage paths.",
        "It is not a release gate and it does not by itself prove unbiasedness.",
        "",
        "## Workload",
        "",
        f"- batch: `{workload['batch']}`",
        f"- q_len: `{workload['q_len']}`",
        f"- k_len: `{workload['k_len']}`",
        f"- d_head: `{workload['d_head']}`",
        f"- q_seed: `{workload['q_seed']}`",
        f"- k_seed: `{workload['k_seed']}`",
        f"- source: `{workload['vector_source']}`",
        "",
        "## Metrics",
        "",
        "| Algorithm | Residual | Mean signed error | Mean abs error | Normalized mean bias | Normalized mean abs error | Error variance | 95th pct abs error |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in algorithms:
        lines.append(
            "| "
            f"`{item['algorithm']}` | `{item['residual_mode']}` | "
            f"{float(item['mean_signed_error']):.4f} | "
            f"{float(item['mean_abs_error']):.4f} | "
            f"{float(item['normalized_mean_bias']):.4f} | "
            f"{float(item['normalized_mean_abs_error']):.4f} | "
            f"{float(item['error_variance']):.4f} | "
            f"{float(item['q95_abs_error']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Honest Takeaway",
            "",
            f"- `paper_prod_qjl` differs from `paper_mse` by a normalized mean-abs-error delta of {float(comparison['normalized_mean_abs_error_delta']):+.4f} on this fixed workload.",
            f"- The signed-bias delta on the same workload is {float(comparison['normalized_mean_bias_delta']):+.4f}.",
            f"- The absolute signed-bias magnitude delta is {float(comparison['normalized_mean_bias_magnitude_delta']):+.4f}.",
            "- These numbers make the two-stage path directly measurable, but they do not justify a theorem-level unbiasedness claim here.",
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
    queries, keys = _synthetic_queries_and_keys(args)

    algorithms = [
        TurboQuantConfig.from_preset("paper_mse"),
        TurboQuantConfig.from_preset("paper_prod"),
    ]
    results = [_score_stats(config, queries, keys) for config in algorithms]

    for result in results:
        for key, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise SystemExit(f"non-finite inner-product bias metric for {key}: {result}")

    by_algorithm = {str(item["algorithm"]): item for item in results}
    baseline = by_algorithm["paper_mse"]
    prod = by_algorithm["paper_prod_qjl"]

    payload = {
        "schema_version": "2",
        "metric_family": "inner_product_bias",
        "status": "ok",
        "environment": collect_environment_metadata(
            model="synthetic-inner-product",
            mode="inner_product_bias",
        ),
        "workload": {
            "batch": args.batch,
            "q_len": args.q_len,
            "k_len": args.k_len,
            "d_head": args.d_head,
            "q_seed": args.q_seed,
            "k_seed": args.k_seed,
            "vector_source": "synthetic",
            "description": (
                "Synthetic rotated-space attention-score workload used to compare "
                "the scalar-only paper_mse path against the paper_prod_qjl two-stage path."
            ),
        },
        "algorithms": results,
        "comparison": {
            "normalized_mean_abs_error_delta": (
                float(prod["normalized_mean_abs_error"])
                - float(baseline["normalized_mean_abs_error"])
            ),
            "normalized_mean_bias_delta": (
                float(prod["normalized_mean_bias"])
                - float(baseline["normalized_mean_bias"])
            ),
            "normalized_mean_bias_magnitude_delta": (
                abs(float(prod["normalized_mean_bias"]))
                - abs(float(baseline["normalized_mean_bias"]))
            ),
        },
        "companion_artifacts": [SUMMARY_JSON, METRICS_CSV, SUMMARY_MD],
        "notes": [
            "This is a research validation metric, not a release gate.",
            "The repo keeps the unbiased-inner-product claim explicitly open until stronger retained evidence exists.",
            "The fixed synthetic workload is intended to keep score-estimation changes reproducible and comparable across commits.",
        ],
    }

    summary_path = output_dir / SUMMARY_JSON
    csv_path = output_dir / METRICS_CSV
    markdown_path = output_dir / SUMMARY_MD

    write_json(summary_path, payload)
    _write_metrics_csv(csv_path, results)
    markdown_path.write_text(_render_markdown_summary(payload), encoding="utf-8")

    print(summary_path)
    print(csv_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
