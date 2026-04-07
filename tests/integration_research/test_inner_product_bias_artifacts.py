from __future__ import annotations

from pathlib import Path

from tests.integration_research.helpers import (
    assert_common_research_fields,
    load_csv_rows,
    load_json,
    run_script,
)


def test_inner_product_bias_artifacts_align(tmp_path: Path) -> None:
    output_dir = tmp_path / "inner_product_bias"
    result = run_script(
        "benchmarks/runtime_cert/run_inner_product_bias_eval.py",
        "--output-dir",
        str(output_dir),
        "--batch",
        "2",
        "--q-len",
        "4",
        "--k-len",
        "4",
        "--d-head",
        "64",
    )

    assert result.returncode == 0, result.stderr or result.stdout

    summary_path = output_dir / "inner_product_bias_summary.json"
    csv_path = output_dir / "inner_product_bias_metrics.csv"
    markdown_path = output_dir / "inner_product_bias_summary.md"
    assert summary_path.is_file()
    assert csv_path.is_file()
    assert markdown_path.is_file()

    payload = load_json(summary_path)
    rows = load_csv_rows(csv_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert_common_research_fields(payload)
    assert payload["metric_family"] == "inner_product_bias"
    assert payload["scope"] == "research-only"
    assert payload["status"] == "ok"
    assert payload["preset"] == "paper_mse_vs_paper_prod_qjl"
    assert payload["family"] == "synthetic"
    assert payload["artifact_paths"]["summary_markdown"] == "inner_product_bias_summary.md"
    assert {
        "normalized_mean_abs_error_delta",
        "normalized_mean_bias_delta",
        "normalized_mean_bias_magnitude_delta",
    }.issubset(payload["comparison"])
    for item in payload["algorithms"]:
        assert {
            "run_id",
            "preset",
            "family",
            "algorithm",
            "residual_mode",
            "normalized_mean_bias",
            "normalized_mean_abs_error",
        }.issubset(item)

    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {payload["run_id"]}
    assert {row["preset"] for row in rows} == {payload["preset"]}
    assert {row["family"] for row in rows} == {payload["family"]}
    assert payload["run_id"] in markdown
    assert f"`{payload['preset']}`" in markdown
    assert f"`{payload['family']}`" in markdown
