from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_HEADER = (
    "<!-- Generated from runtime-cert artifacts by "
    "scripts/render_benchmark_snapshot.py. Do not edit by hand. -->"
)
JsonDict = dict[str, Any]


def _read_json(path: Path) -> JsonDict:
    return cast(JsonDict, json.loads(path.read_text(encoding="utf-8")))


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_mib(value: float) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def _format_ratio(dense_value: float, tq_value: float) -> str:
    if tq_value <= 0:
        return "n/a"
    return f"{dense_value / tq_value:.1f}x smaller"


def _slug_from_artifact_dir(artifact_dir: Path) -> str:
    return f"BENCHMARK_SNAPSHOT_{artifact_dir.name}.md"


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_by_key(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["model"], row["prompt_class"], row["mode"])
        grouped.setdefault(key, []).append(row)
    return grouped


def _avg_float(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return sum(values) / len(values)


def _render_table(rows: list[dict[str, str]]) -> list[str]:
    grouped = _rows_by_key(rows)
    models = sorted({row["model"] for row in rows})
    prompt_classes = ["short", "medium", "long"]

    lines = [
        "| Model | Prompt class | Dense peak | TurboQuant peak | Memory reduction | Dense TPS | TurboQuant TPS | Throughput delta |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        for prompt_class in prompt_classes:
            dense = grouped.get((model, prompt_class, "dense"), [])
            tq = grouped.get((model, prompt_class, "turboquant"), [])
            if dense is None or tq is None:
                continue
            if not dense or not tq:
                continue

            dense_peak = _avg_float(dense, "peak_memory_bytes")
            tq_peak = _avg_float(tq, "peak_memory_bytes")
            dense_tps = _avg_float(dense, "tokens_per_second")
            tq_tps = _avg_float(tq, "tokens_per_second")
            reduction_pct = (1 - tq_peak / dense_peak) * 100 if dense_peak > 0 else 0.0
            delta_pct = (tq_tps / dense_tps - 1) * 100 if dense_tps > 0 else 0.0
            lines.append(
                "| "
                f"`{model}` | `{prompt_class}` | {_format_mib(dense_peak)} | "
                f"{_format_mib(tq_peak)} | {reduction_pct:.1f}% ({_format_ratio(dense_peak, tq_peak)}) | "
                f"{dense_tps:.2f} | {tq_tps:.2f} | {delta_pct:.1f}% |"
            )
    return lines


def _render_inner_product_bias_section(
    summary: JsonDict | None,
    *,
    expected_commit: str,
) -> list[str]:
    if summary is None:
        return []

    environment = cast(JsonDict, summary.get("environment", {}))
    bias_commit = str(environment.get("git_commit", "unknown"))
    if bias_commit != expected_commit:
        return []

    algorithms = cast(list[JsonDict], summary.get("algorithms", []))
    if not algorithms:
        return []

    comparison = cast(JsonDict, summary.get("comparison", {}))
    lines = [
        "",
        "## Inner-Product Bias Snapshot",
        "",
        "This artifact also includes a research-only synthetic score comparison for the paper-facing scalar-only and two-stage paths.",
        "It is evidence about current estimator behavior, not a release gate and not yet a proof of unbiasedness.",
        "",
        "| Algorithm | Residual mode | Mean signed error | Mean abs error | Normalized mean bias | Normalized mean abs error | Error variance | 95th pct abs error |",
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
            "Bias snapshot takeaway:",
            f"The current `paper_prod_qjl` path differs from `paper_mse` by a normalized mean-abs-error delta of {float(comparison.get('normalized_mean_abs_error_delta', 0.0)):+.4f} on this fixed synthetic workload.",
            "That makes the two-stage path directly measurable, but it does not by itself justify an unbiasedness claim yet.",
        ]
    )
    return lines


def _render_kv_paper_eval_section(summary: JsonDict | None) -> list[str]:
    if summary is None:
        return []

    stages = cast(list[JsonDict], summary.get("stages", []))
    if not stages:
        return []

    lines = [
        "",
        "## KV Research Rollup",
        "",
        "This snapshot also records the retained unified KV research bundle when one is supplied.",
        "It complements the certification sweep; it does not replace the product release gate.",
        "",
        "| Stage | Tier | Status | Notes |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for stage in stages:
        note = "; ".join(cast(list[str], stage.get("notes", []))) or "-"
        lines.append(
            f"| {stage['name']} | `{stage['tier']}` | `{stage['status']}` | {note} |"
        )

    tier_counts = cast(JsonDict, summary.get("tier_counts", {}))
    heavy = cast(JsonDict, tier_counts.get("heavy-offline", {}))
    fast = cast(JsonDict, tier_counts.get("fast-check", {}))
    lines.extend(
        [
            "",
            "KV rollup takeaway:",
            f"The retained bundle executed {int(fast.get('executed', 0))} fast-check stages and {int(heavy.get('executed', 0))} heavy-offline stages.",
            "Any Gemma paper_mse tranche reported here remains observational research unless and until the repo promotes it into a symmetric guardrail.",
        ]
    )
    return lines


def _render_vector_search_section(summary: JsonDict | None) -> list[str]:
    if summary is None:
        return []

    evaluations = cast(list[JsonDict], summary.get("evaluations", []))
    if not evaluations:
        return []

    dataset = cast(JsonDict, summary.get("dataset", {}))
    lines = [
        "",
        "## Vector-Search Research Snapshot",
        "",
        "This snapshot can also carry an optional research-only retrieval rollup.",
        "It remains outside the supported Apple-MLX product contract.",
        "",
        f"Dataset selector: `{dataset.get('selector', 'unknown')}` with `{dataset.get('doc_count', 0)}` docs and `{dataset.get('query_count', 0)}` queries.",
        "",
        "| Method | Classification | Recall@1 | Recall@3 | Compression ratio | Query ms/query |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in evaluations:
        lines.append(
            "| "
            f"`{row['method']}` | {row['classification']} | "
            f"{float(row['recall_at_1']):.3f} | {float(row['recall_at_3']):.3f} | "
            f"{float(row['compression_ratio']):.2f}x | {float(row['query_ms_per_query']):.2f} |"
        )

    lines.extend(
        [
            "",
            "Vector-search takeaway:",
            "These retrieval metrics are retained as research context only. They do not promote vector search into the release-gated product surface.",
        ]
    )
    return lines


def render_snapshot(
    artifact_dir: Path,
    artifact_uri: str,
    *,
    kv_paper_eval_summary: Path | None = None,
    vector_search_summary: Path | None = None,
) -> str:
    summary = _read_json(artifact_dir / "certification_summary.json")
    manifest = _read_json(artifact_dir / "cert_manifest.json")
    preflight = _read_json(artifact_dir / "preflight.json")
    rows = _load_rows(artifact_dir / "aggregate_runs.csv")
    inner_product_bias = _read_optional_json(
        artifact_dir / "inner_product_bias_summary.json"
    )
    kv_rollup = _read_optional_json(kv_paper_eval_summary) if kv_paper_eval_summary else None
    vector_rollup = (
        _read_optional_json(vector_search_summary) if vector_search_summary else None
    )

    memory_deltas = cast(list[JsonDict], summary["memory_deltas"])
    speed_deltas = cast(list[JsonDict], summary["speed_deltas"])
    min_memory_reduction = min(float(item["reduction_pct"]) for item in memory_deltas)
    max_memory_reduction = max(float(item["reduction_pct"]) for item in memory_deltas)
    worst_speed_delta = min(float(item["delta_pct"]) for item in speed_deltas)
    best_speed_delta = max(float(item["delta_pct"]) for item in speed_deltas)

    commit = next((row["commit"] for row in rows if row.get("commit")), "unknown")
    models = cast(list[str], summary["models"])
    prompt_classes = cast(list[str], summary["prompt_classes"])
    manifest_sha = _sha256(artifact_dir / "cert_manifest.json")
    platform = cast(str, manifest["platform"])
    families = cast(list[str], manifest["certification_scope"]["families"])

    lines = [
        GENERATED_HEADER,
        f"# Benchmark Snapshot {artifact_dir.name}",
        "",
        "This is a dated benchmark snapshot, not a timeless product claim.",
        "It summarizes the dense-vs-TurboQuant sweep retained in one runtime-cert evidence bundle.",
        "",
        "## What This Helps With",
        "",
        "TurboQuant helps when KV-cache memory pressure is the bottleneck on the Apple-Silicon MLX path.",
        f"In this run, peak benchmark memory dropped by {min_memory_reduction:.1f}% to {max_memory_reduction:.1f}% across the allowlisted Llama and Gemma sweeps.",
        "That makes it useful for fitting longer prompts or reducing KV footprint on the supported runtime path.",
        "",
        "It does not help raw decode throughput in the current uncompiled path.",
        f"In this same run, TurboQuant throughput was {abs(best_speed_delta):.1f}% to {abs(worst_speed_delta):.1f}% lower than dense baselines.",
        "If your bottleneck is tokens-per-second or per-token latency rather than memory, these numbers argue against using it as a speed optimization.",
        "",
        "## Honest Takeaway",
        "",
        "On this commit and hardware, TurboQuant behaves like a memory-saving tradeoff rather than a speedup.",
        "The strongest honest claim from this snapshot is:",
        "",
        f"- It preserved the supported Apple runtime path for `{', '.join(families)}` with a `PASS` manifest on `{platform}`.",
        f"- It cut measured peak benchmark memory by roughly {min_memory_reduction:.1f}% to {max_memory_reduction:.1f}%.",
        f"- It reduced measured decode throughput by roughly {abs(best_speed_delta):.1f}% to {abs(worst_speed_delta):.1f}%.",
        "- It is therefore a memory-footprint tool first, not a throughput benchmark winner.",
        "",
        "## Sweep Summary",
        "",
    ]
    lines.extend(_render_table(rows))
    lines.extend(
        _render_inner_product_bias_section(
            inner_product_bias,
            expected_commit=commit,
        )
    )
    lines.extend(_render_kv_paper_eval_section(kv_rollup))
    lines.extend(_render_vector_search_section(vector_rollup))
    lines.extend(
        [
            "",
            "## Scope And Limits",
            "",
            f"- Commit under test: `{commit}`",
            f"- Models: {', '.join(f'`{model}`' for model in models)}",
            f"- Prompt classes: {', '.join(f'`{item}`' for item in prompt_classes)}",
            "- Decode length: `64` new tokens per paired run",
            "- Benchmark mode: paired `dense` vs `turboquant` sweeps",
            "- This snapshot does not prove a universal speedup, broad model-family support, or non-Apple portability.",
            "- The quality stages in the certification bundle are guardrails against catastrophic regressions; they are not evidence that TurboQuant improves model quality.",
            "",
            "## Provenance",
            "",
            f"- `artifact_uri_or_manifest_digest`: `{artifact_uri}` and `sha256:{manifest_sha}`",
            f"- `git_commit`: `{commit}`",
            f"- `model_ids`: {', '.join(f'`{model}`' for model in models)}",
            f"- `mlx_version`: `{preflight['mlx_version']}`",
            f"- `hardware`: `{preflight['platform']}`",
            "- `script`: `bash scripts/certify_apple_runtime.sh`",
            "- `args`: certification script invoked `benchmarks/runtime_cert/run_dense_vs_tq.py` for each model with `--prompt-file benchmarks/runtime_cert/prompts/{short,medium,long}.jsonl --max-new-tokens 64 --seed 42 --mode both`",
            "- `extra_research_metrics`: optional `benchmarks/runtime_cert/run_inner_product_bias_eval.py` outputs (`inner_product_bias_summary.json`, `inner_product_bias_metrics.csv`, `inner_product_bias_summary.md`) when retained in the same artifact directory, plus optional `kv_paper_eval_summary.json` and `vector_search_summary.json` sidecars when supplied to the renderer",
            "",
            "## Addressable Evidence",
            "",
            f"- Local artifact directory: `{artifact_dir}`",
            f"- Local portable artifact: `{artifact_dir}.zip`",
            f"- Manifest: `{artifact_dir / 'cert_manifest.json'}`",
            "- Hosted GitHub Actions evidence is still preferable for release-facing publication once a self-hosted Apple runner completes the queued workflow run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a dated benchmark snapshot from a runtime-cert artifact.")
    parser.add_argument("--artifact-dir", required=True, help="Path to a runtime-cert artifact directory.")
    parser.add_argument(
        "--artifact-uri",
        default="",
        help="Artifact URI or manifest digest label to record in the provenance block.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output Markdown path. Defaults to docs/history/BENCHMARK_SNAPSHOT_<timestamp>.md",
    )
    parser.add_argument(
        "--kv-paper-eval-summary",
        default="",
        help="Optional path to a kv_paper_eval_summary.json sidecar to roll into the snapshot.",
    )
    parser.add_argument(
        "--vector-search-summary",
        default="",
        help="Optional path to a vector_search_summary.json sidecar to roll into the snapshot.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    if not artifact_dir.is_dir():
        raise SystemExit(f"artifact-dir does not exist: {artifact_dir}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else REPO_ROOT / "docs" / "history" / _slug_from_artifact_dir(artifact_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_uri = args.artifact_uri or str(artifact_dir)
    content = render_snapshot(
        artifact_dir,
        artifact_uri,
        kv_paper_eval_summary=(
            Path(args.kv_paper_eval_summary).resolve()
            if args.kv_paper_eval_summary
            else None
        ),
        vector_search_summary=(
            Path(args.vector_search_summary).resolve()
            if args.vector_search_summary
            else None
        ),
    )
    output_path.write_text(content, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
