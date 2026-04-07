from __future__ import annotations

from pathlib import Path

from tests.integration_research.helpers import (
    assert_common_research_fields,
    load_json,
    run_script,
)


def test_vector_search_mini_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "vector_search"
    result = run_script(
        "benchmarks/vector_search/run_vector_search_eval.py",
        "--output-dir",
        str(output_dir),
        "--dataset",
        "mini",
        "--embedding-dim",
        "64",
        "--presets",
        "paper_mse",
    )

    assert result.returncode == 0, result.stderr or result.stdout

    summary_path = output_dir / "vector_search_summary.json"
    csv_path = output_dir / "vector_search_metrics.csv"
    markdown_path = output_dir / "vector_search_summary.md"
    assert summary_path.is_file()
    assert csv_path.is_file()
    assert markdown_path.is_file()

    payload = load_json(summary_path)
    assert_common_research_fields(payload)
    assert payload["metric_family"] == "vector_search"
    assert payload["scope"] == "research-only"
    assert payload["status"] == "ok"
    assert payload["preset"] == "paper_mse"
    assert payload["family"] == "not-applicable"
    assert payload["support_scope"] == "research-only"
    assert payload["dataset"]["selector"] == "mini"
    assert payload["artifact_paths"]["metrics_csv"] == "vector_search_metrics.csv"
    assert len(payload["evaluations"]) == 2
    for row in payload["evaluations"]:
        assert {
            "run_id",
            "family",
            "method",
            "classification",
            "recall_at_1",
            "recall_at_3",
            "index_bytes",
        }.issubset(row)
