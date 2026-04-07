from __future__ import annotations

from pathlib import Path

from tests.integration_research.helpers import assert_common_research_fields, load_json, run_script


def test_kv_paper_eval_fast_check_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "kv_paper_eval"
    result = run_script(
        "benchmarks/runtime_cert/run_kv_paper_eval.py",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr or result.stdout

    summary_path = output_dir / "kv_paper_eval_summary.json"
    markdown_path = output_dir / "kv_paper_eval_summary.md"
    assert summary_path.is_file()
    assert markdown_path.is_file()

    payload = load_json(summary_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert_common_research_fields(payload)
    assert payload["metric_family"] == "kv_paper_eval"
    assert payload["scope"] == "research-only"
    assert payload["preset"] == "mixed"
    assert payload["family"] == "not-configured"
    assert payload["status"] == "passed"
    assert payload["artifact_paths"]["summary_markdown"] == "kv_paper_eval_summary.md"
    assert payload["heavy_offline_requested"] is False
    assert payload["tier_counts"]["fast-check"]["executed"] == 5
    assert payload["tier_counts"]["heavy-offline"]["executed"] == 0

    stages = {stage["stage_id"]: stage for stage in payload["stages"]}
    assert stages["heavy_offline_bundle"]["status"] == "not_requested"
    assert payload["run_id"] in markdown
    assert "Heavy-offline stages are reported explicitly" in markdown
