from __future__ import annotations

from pathlib import Path

from tests.helpers.mlx_env import MLX_SKIP_MARKER
from tests.integration_research.helpers import (
    assert_common_research_fields,
    load_json,
    run_script,
)

pytestmark = MLX_SKIP_MARKER


def test_bit_budget_sweep_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "bit_budget"
    result = run_script(
        "benchmarks/runtime_cert/run_bit_budget_sweep.py",
        "--output-dir",
        str(output_dir),
        "--presets",
        "paper_mse",
        "paper_prod_qjl",
        "--k-bits",
        "3",
        "--batch",
        "1",
        "--tokens",
        "32",
        "--latency-reps",
        "1",
        "--score-batch",
        "2",
        "--score-q-len",
        "4",
        "--score-k-len",
        "4",
        "--d-head",
        "64",
    )

    assert result.returncode == 0, result.stderr or result.stdout

    summary_path = output_dir / "bit_budget_sweep_summary.json"
    assert summary_path.is_file()

    payload = load_json(summary_path)
    assert_common_research_fields(payload)
    assert payload["metric_family"] == "bit_budget_sweep"
    assert payload["scope"] == "research-only"
    assert payload["status"] == "ok"
    assert payload["preset"] == "mixed"
    assert payload["family"] == "synthetic"
    assert payload["support_scope"] == "research-only"
    assert "operating_points" in payload
    assert "report_schema" in payload
    assert "workload" in payload
    assert payload["artifact_paths"]["summary_json"] == "bit_budget_sweep_summary.json"
    ids = {row["operating_point_id"] for row in payload["operating_points"]}
    assert ids == {
        "paper_mse_k3_v4_g64",
        "paper_prod_qjl_k3_v4_g64",
    }
