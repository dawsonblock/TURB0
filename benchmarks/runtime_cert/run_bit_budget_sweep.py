#!/usr/bin/env python3
# flake8: noqa: E402
"""Run a research-only bit-budget sweep on synthetic KV workloads.

This driver keeps the supported product contract unchanged. It uses a fixed
synthetic cache workload plus a fixed synthetic score workload so paper-facing
and research-only preset surfaces can be compared across operating points.

Outputs:

- `bit_budget_sweep_summary.json`
- `bit_budget_sweep_metrics.csv`
- `bit_budget_sweep_summary.md`

The downstream-quality field is retained in the schema, but defaults to null in
this synthetic sweep. Real quality evidence remains available through the
runtime-cert quality lanes and should not be inferred from this script alone.
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
from mlx_lm.models.cache import KVCache

from benchmarks.runtime_cert.research_report_schema import (
    build_artifact_paths,
    build_research_report,
    build_run_id,
)
from benchmarks.runtime_cert.utils import collect_environment_metadata, ensure_artifact_dir, write_json
from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.core.rotation import FixedRotation
from turboquant.runtime.attention import score_block
from turboquant.runtime.kv_interface import TurboQuantKVCache

SUMMARY_JSON = "bit_budget_sweep_summary.json"
METRICS_CSV = "bit_budget_sweep_metrics.csv"
SUMMARY_MD = "bit_budget_sweep_summary.md"
DEFAULT_PRESETS = ("paper_mse", "paper_prod_qjl", "polarquant_exp")
DEFAULT_K_BITS = (4, 3, 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a research-only bit-budget sweep on synthetic KV workloads."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the bit-budget sweep artifacts.",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(DEFAULT_PRESETS),
        help="Preset names to include in the sweep.",
    )
    parser.add_argument(
        "--k-bits",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_BITS),
        help="K-bit settings to sweep for each preset.",
    )
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--k-group-size", type=int, default=64)
    parser.add_argument("--v-group-size", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--d-head", type=int, default=128)
    parser.add_argument("--cache-seed", type=int, default=7)
    parser.add_argument("--latency-reps", type=int, default=5)
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--score-q-len", type=int, default=12)
    parser.add_argument("--score-k-len", type=int, default=16)
    parser.add_argument("--score-q-seed", type=int, default=0)
    parser.add_argument("--score-k-seed", type=int, default=1)
    return parser.parse_args()


def _build_config(
    preset: str,
    *,
    k_bits: int,
    v_bits: int,
    k_group_size: int,
    v_group_size: int,
) -> TurboQuantConfig:
    base = TurboQuantConfig.from_preset(preset)
    kwargs = base.to_state_dict()
    kwargs.update(
        {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "k_group_size": k_group_size,
            "v_group_size": v_group_size,
        }
    )
    cfg = TurboQuantConfig(**kwargs)
    cfg.validate()
    return cfg


def _cache_workload(args: argparse.Namespace) -> tuple[mx.array, mx.array, list[mx.array], list[mx.array]]:
    cache_keys = mx.random.normal(
        shape=(args.batch, args.kv_heads, args.tokens, args.d_head),
        dtype=mx.float16,
        key=mx.random.key(args.cache_seed),
    )
    cache_values = mx.random.normal(
        shape=(args.batch, args.kv_heads, args.tokens, args.d_head),
        dtype=mx.float16,
        key=mx.random.key(args.cache_seed + 1),
    )
    step_keys = [
        mx.random.normal(
            shape=(args.batch, args.kv_heads, 1, args.d_head),
            dtype=mx.float16,
            key=mx.random.key(args.cache_seed + 2 + index * 2),
        )
        for index in range(args.latency_reps)
    ]
    step_values = [
        mx.random.normal(
            shape=(args.batch, args.kv_heads, 1, args.d_head),
            dtype=mx.float16,
            key=mx.random.key(args.cache_seed + 3 + index * 2),
        )
        for index in range(args.latency_reps)
    ]
    mx.eval(cache_keys, cache_values, *step_keys, *step_values)
    return cache_keys, cache_values, step_keys, step_values


def _score_workload(args: argparse.Namespace) -> tuple[mx.array, mx.array]:
    queries = mx.random.normal(
        shape=(args.score_batch, args.score_q_len, args.d_head),
        key=mx.random.key(args.score_q_seed),
    )
    keys = mx.random.normal(
        shape=(args.score_batch, args.score_k_len, args.d_head),
        key=mx.random.key(args.score_k_seed),
    )
    return queries, keys


def _eval_dense_cache(cache: KVCache) -> None:
    values = []
    for attr in ("keys", "values"):
        arr = getattr(cache, attr, None)
        if arr is not None:
            values.append(arr)
    if values:
        mx.eval(*values)


def _eval_tq_cache(cache: TurboQuantKVCache) -> None:
    values = []
    if cache.k_packed is not None:
        values.append(cache.k_packed)
    if cache.storage_mode == "paper_kv" and cache.v_blocks:
        last_v = cache.v_blocks[-1]
        if last_v.packed_main is not None:
            values.append(last_v.packed_main)
    if values:
        mx.eval(*values)


def _measure_latency_ms(
    config: TurboQuantConfig,
    cache_keys: mx.array,
    cache_values: mx.array,
    step_keys: list[mx.array],
    step_values: list[mx.array],
) -> tuple[float, float, float]:
    dense_cache = KVCache()
    dense_cache.update_and_fetch(cache_keys, cache_values)
    _eval_dense_cache(dense_cache)
    t0 = time.perf_counter()
    for key_step, value_step in zip(step_keys, step_values):
        dense_cache.update_and_fetch(key_step, value_step)
        _eval_dense_cache(dense_cache)
    dense_ms = ((time.perf_counter() - t0) / len(step_keys)) * 1000.0

    tq_cache = TurboQuantKVCache(config)
    tq_cache.update_and_fetch(cache_keys, cache_values)
    _eval_tq_cache(tq_cache)
    t0 = time.perf_counter()
    for key_step, value_step in zip(step_keys, step_values):
        tq_cache.update_and_fetch(key_step, value_step)
        _eval_tq_cache(tq_cache)
    tq_ms = ((time.perf_counter() - t0) / len(step_keys)) * 1000.0

    overhead_pct = ((tq_ms / dense_ms) - 1.0) * 100.0 if dense_ms > 0 else 0.0
    return dense_ms, tq_ms, overhead_pct


def _distortion_metrics(config: TurboQuantConfig, cache_keys: mx.array) -> dict[str, float]:
    pipe = TurboQuantPipeline(config)
    block = pipe.encode_k(cache_keys.astype(mx.float32))
    decoded = pipe.decode_k(block).astype(mx.float32)
    original = cache_keys.astype(mx.float32)
    diff = decoded - original
    rel_abs = float(
        (mx.mean(mx.abs(diff)) / (mx.mean(mx.abs(original)) + 1e-6)).item()
    )
    mse = float(mx.mean(diff * diff).item())
    return {
        "key_relative_abs_error": rel_abs,
        "key_mse": mse,
        "compressed_key_bytes": float(block.byte_size()),
    }


def _true_scores(config: TurboQuantConfig, q: mx.array, k: mx.array) -> mx.array:
    rotation = FixedRotation.from_config(config, int(q.shape[-1]))
    q_rot = rotation.apply(q.astype(mx.float32))
    k_rot = rotation.apply(k.astype(mx.float32))
    return q_rot @ mx.swapaxes(k_rot, -1, -2)


def _bias_metrics(config: TurboQuantConfig, q: mx.array, k: mx.array) -> dict[str, float]:
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
    return {
        "normalized_mean_bias": float(mx.mean(flat_error).item()) / mean_abs_true,
        "normalized_mean_abs_error": float(mx.mean(flat_abs_error).item())
        / mean_abs_true,
        "bias_error_variance": float(mx.var(flat_error).item()),
    }


def _operating_point_id(preset: str, k_bits: int, v_bits: int, group_size: int) -> str:
    return f"{preset}_k{k_bits}_v{v_bits}_g{group_size}"


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "family",
        "operating_point_id",
        "preset",
        "classification",
        "algorithm",
        "storage_mode",
        "configured_k_bits",
        "configured_v_bits",
        "k_group_size",
        "v_group_size",
        "dense_total_bytes",
        "compressed_total_bytes",
        "measured_effective_bits_per_channel_total",
        "measured_compression_ratio",
        "memory_reduction_pct",
        "key_relative_abs_error",
        "key_mse",
        "normalized_mean_bias",
        "normalized_mean_abs_error",
        "bias_error_variance",
        "dense_latency_ms",
        "turboquant_latency_ms",
        "latency_overhead_pct",
        "downstream_quality_available",
        "downstream_quality_metric_name",
        "downstream_quality_metric_value",
        "downstream_quality_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_summary(payload: dict[str, Any]) -> str:
    rows = payload["operating_points"]
    best_memory = max(rows, key=lambda row: float(row["memory_reduction_pct"]))
    best_latency = min(rows, key=lambda row: float(row["latency_overhead_pct"]))

    lines = [
        "# Bit-Budget Sweep Summary",
        "",
        "This is a research-only synthetic sweep across explicit preset and bit-budget operating points.",
        "It does not widen the supported product contract and it does not replace real-model quality evidence.",
        "",
        "## Scope",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- preset group: `{payload['preset']}`",
        f"- family: `{payload['family']}`",
        f"- support scope: `{payload['support_scope']}`",
        f"- presets: {', '.join(f'`{preset}`' for preset in payload['presets'])}",
        f"- k_bits sweep: {', '.join(f'`{bits}`' for bits in payload['k_bits'])}",
        "",
        "## Workload",
        "",
        f"- cache workload: batch=`{payload['workload']['cache']['batch']}`, kv_heads=`{payload['workload']['cache']['kv_heads']}`, tokens=`{payload['workload']['cache']['tokens']}`, d_head=`{payload['workload']['cache']['d_head']}`",
        f"- score workload: batch=`{payload['workload']['score']['batch']}`, q_len=`{payload['workload']['score']['q_len']}`, k_len=`{payload['workload']['score']['k_len']}`, d_head=`{payload['workload']['score']['d_head']}`",
        "",
        "## Operating Points",
        "",
        "| ID | Preset | Classification | Compression ratio | Memory reduction | Distortion (rel abs) | Bias (norm abs err) | Latency overhead | Downstream quality |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operating_point_id']}` | `{row['preset']}` | {row['classification']} | "
            f"{float(row['measured_compression_ratio']):.2f}x | "
            f"{float(row['memory_reduction_pct']):.1f}% | "
            f"{float(row['key_relative_abs_error']):.4f} | "
            f"{float(row['normalized_mean_abs_error']):.4f} | "
            f"{float(row['latency_overhead_pct']):+.1f}% | "
            f"{row['downstream_quality_note']} |"
        )

    lines.extend(
        [
            "",
            "## Honest Takeaways",
            "",
            f"- Highest measured memory reduction in this sweep came from `{best_memory['operating_point_id']}` at {float(best_memory['memory_reduction_pct']):.1f}%.",
            f"- Lowest measured latency overhead in this sweep came from `{best_latency['operating_point_id']}` at {float(best_latency['latency_overhead_pct']):+.1f}% relative to dense cache updates.",
            "- The downstream-quality field is present in the schema but intentionally null in this synthetic sweep; use runtime-cert quality artifacts for real-model guardrails.",
            "- None of these rows, by themselves, prove product support expansion or theorem-level paper equivalence.",
            "",
            "## Companion Artifacts",
            "",
            f"- `{SUMMARY_JSON}` — structured sweep summary",
            f"- `{METRICS_CSV}` — row-oriented operating-point table",
            f"- `{SUMMARY_MD}` — human-readable summary",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    output_dir = ensure_artifact_dir(args.output_dir)
    environment = collect_environment_metadata(
        model="synthetic-kv-workload",
        mode="bit_budget_sweep",
    )
    run_id = build_run_id(
        timestamp=str(environment["timestamp"]),
        label=f"{'_'.join(args.presets)}_{'_'.join(str(bits) for bits in args.k_bits)}",
        mode="bit_budget_sweep",
    )

    cache_keys, cache_values, step_keys, step_values = _cache_workload(args)
    score_queries, score_keys = _score_workload(args)
    dense_total_bytes = int(cache_keys.nbytes + cache_values.nbytes)
    total_channels = (
        args.batch * args.kv_heads * args.tokens * args.d_head * 2
    )

    rows: list[dict[str, Any]] = []
    for preset in args.presets:
        preset_meta = TurboQuantConfig.preset_metadata(preset)
        for k_bits in args.k_bits:
            config = _build_config(
                preset,
                k_bits=k_bits,
                v_bits=args.v_bits,
                k_group_size=args.k_group_size,
                v_group_size=args.v_group_size,
            )
            cache = TurboQuantKVCache(config)
            cache.update_and_fetch(cache_keys, cache_values)
            _eval_tq_cache(cache)
            compressed_total_bytes = int(cache.nbytes)
            measured_bpc = (compressed_total_bytes * 8.0) / float(total_channels)
            compression_ratio = (
                dense_total_bytes / compressed_total_bytes
                if compressed_total_bytes > 0
                else float("inf")
            )
            memory_reduction_pct = (
                (1.0 - (compressed_total_bytes / dense_total_bytes)) * 100.0
                if dense_total_bytes > 0
                else 0.0
            )

            distortion = _distortion_metrics(config, cache_keys)
            bias = _bias_metrics(config, score_queries, score_keys)
            dense_ms, tq_ms, latency_overhead = _measure_latency_ms(
                config,
                cache_keys,
                cache_values,
                step_keys,
                step_values,
            )

            row = {
                "run_id": run_id,
                "family": "synthetic",
                "operating_point_id": _operating_point_id(
                    preset,
                    k_bits,
                    args.v_bits,
                    args.k_group_size,
                ),
                "preset": preset,
                "classification": preset_meta["classification"],
                "algorithm": config.algorithm_family(),
                "storage_mode": cache.storage_mode,
                "configured_k_bits": config.k_bits,
                "configured_v_bits": config.v_bits,
                "k_group_size": config.k_group_size,
                "v_group_size": config.v_group_size,
                "dense_total_bytes": dense_total_bytes,
                "compressed_total_bytes": compressed_total_bytes,
                "measured_effective_bits_per_channel_total": measured_bpc,
                "measured_compression_ratio": compression_ratio,
                "memory_reduction_pct": memory_reduction_pct,
                "key_relative_abs_error": distortion["key_relative_abs_error"],
                "key_mse": distortion["key_mse"],
                "normalized_mean_bias": bias["normalized_mean_bias"],
                "normalized_mean_abs_error": bias["normalized_mean_abs_error"],
                "bias_error_variance": bias["bias_error_variance"],
                "dense_latency_ms": dense_ms,
                "turboquant_latency_ms": tq_ms,
                "latency_overhead_pct": latency_overhead,
                "downstream_quality_available": False,
                "downstream_quality_metric_name": None,
                "downstream_quality_metric_value": None,
                "downstream_quality_note": (
                    "not available in synthetic sweep; use runtime-cert quality artifacts"
                ),
            }

            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise SystemExit(
                        f"non-finite sweep metric for preset {preset!r} at k_bits={k_bits}: {row}"
                    )

            rows.append(row)

    artifact_paths = build_artifact_paths(
        summary_json=SUMMARY_JSON,
        metrics_csv=METRICS_CSV,
        summary_markdown=SUMMARY_MD,
    )
    notes = [
        "This is a research-only synthetic sweep, not a release gate.",
        "The downstream-quality field remains null here because this command does not run real-model long-context or teacher-forcing quality evaluation.",
        "Use the runtime-cert quality artifacts for supported-family quality guardrails.",
    ]
    metrics = {
        "operating_point_count": len(rows),
        "best_memory_reduction_pct": max(
            float(row["memory_reduction_pct"]) for row in rows
        ),
        "lowest_latency_overhead_pct": min(
            float(row["latency_overhead_pct"]) for row in rows
        ),
    }

    payload = build_research_report(
        schema_version="1",
        metric_family="bit_budget_sweep",
        run_id=run_id,
        environment=environment,
        preset=args.presets[0] if len(args.presets) == 1 else "mixed",
        family="synthetic",
        scope="research-only",
        mode="bit_budget_sweep",
        status="ok",
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes=notes,
        support_scope="research-only",
        presets=list(args.presets),
        k_bits=list(args.k_bits),
        workload={
            "cache": {
                "batch": args.batch,
                "kv_heads": args.kv_heads,
                "tokens": args.tokens,
                "d_head": args.d_head,
                "seed": args.cache_seed,
                "dtype": "float16",
                "vector_source": "synthetic",
            },
            "score": {
                "batch": args.score_batch,
                "q_len": args.score_q_len,
                "k_len": args.score_k_len,
                "d_head": args.d_head,
                "q_seed": args.score_q_seed,
                "k_seed": args.score_k_seed,
                "vector_source": "synthetic",
            },
        },
        report_schema={
            "configured_fields": [
                "preset",
                "classification",
                "configured_k_bits",
                "configured_v_bits",
                "k_group_size",
                "v_group_size",
            ],
            "measured_fields": [
                "measured_effective_bits_per_channel_total",
                "measured_compression_ratio",
                "memory_reduction_pct",
                "key_relative_abs_error",
                "key_mse",
                "normalized_mean_bias",
                "normalized_mean_abs_error",
                "bias_error_variance",
                "dense_latency_ms",
                "turboquant_latency_ms",
                "latency_overhead_pct",
            ],
            "optional_fields": [
                "downstream_quality_metric_name",
                "downstream_quality_metric_value",
                "downstream_quality_note",
            ],
        },
        operating_points=rows,
        companion_artifacts=[SUMMARY_JSON, METRICS_CSV, SUMMARY_MD],
    )

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
