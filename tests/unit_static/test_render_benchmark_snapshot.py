from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.render_benchmark_snapshot import render_snapshot


def test_render_snapshot_includes_optional_research_sidecars(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "20260407_014958"
    artifact_dir.mkdir()

    (artifact_dir / "certification_summary.json").write_text(
        json.dumps(
            {
                "memory_deltas": [{"reduction_pct": 90.0}],
                "speed_deltas": [{"delta_pct": -95.0}],
                "models": ["demo-model"],
                "prompt_classes": ["short"],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "cert_manifest.json").write_text(
        json.dumps(
            {
                "platform": "darwin-arm64",
                "certification_scope": {"families": ["llama", "gemma"]},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "preflight.json").write_text(
        json.dumps({"mlx_version": "0.31.1", "platform": "macOS-26.2-arm64"}),
        encoding="utf-8",
    )

    with (artifact_dir / "aggregate_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "commit",
                "model",
                "prompt_class",
                "mode",
                "peak_memory_bytes",
                "tokens_per_second",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "commit": "e86184b",
                "model": "demo-model",
                "prompt_class": "short",
                "mode": "dense",
                "peak_memory_bytes": "1024",
                "tokens_per_second": "100.0",
            }
        )
        writer.writerow(
            {
                "commit": "e86184b",
                "model": "demo-model",
                "prompt_class": "short",
                "mode": "turboquant",
                "peak_memory_bytes": "128",
                "tokens_per_second": "5.0",
            }
        )

    kv_summary = tmp_path / "kv_paper_eval_summary.json"
    kv_summary.write_text(
        json.dumps(
            {
                "tier_counts": {
                    "fast-check": {"executed": 5},
                    "heavy-offline": {"executed": 4},
                },
                "stages": [
                    {
                        "name": "paper_mse quality evaluation (gemma research-only)",
                        "tier": "heavy-offline",
                        "status": "captured",
                        "notes": [
                            "Research-only observational tranche; this does not promote Gemma to a symmetric release guardrail."
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    vector_summary = tmp_path / "vector_search_summary.json"
    vector_summary.write_text(
        json.dumps(
            {
                "dataset": {"selector": "mini", "doc_count": 12, "query_count": 7},
                "evaluations": [
                    {
                        "method": "paper_mse",
                        "classification": "paper-facing",
                        "recall_at_1": 1.0,
                        "recall_at_3": 1.0,
                        "compression_ratio": 8.5,
                        "query_ms_per_query": 0.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    content = render_snapshot(
        artifact_dir,
        "artifact-ref",
        kv_paper_eval_summary=kv_summary,
        vector_search_summary=vector_summary,
    )

    assert "## KV Research Rollup" in content
    assert "paper_mse quality evaluation (gemma research-only)" in content
    assert "## Vector-Search Research Snapshot" in content
    assert "Dataset selector: `mini`" in content
