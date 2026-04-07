import json
from pathlib import Path

from benchmarks.runtime_cert.collect_metrics import collect_run_artifacts


def test_collect_run_artifacts_ignores_non_run_json(tmp_path: Path) -> None:
    run_record = {
        "run_id": "demo-run",
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mode": "dense",
        "prompt_id": "short_001",
        "prompt_class": "short",
        "status": "ok",
    }
    (tmp_path / "dense_run.json").write_text(
        json.dumps(run_record),
        encoding="utf-8",
    )

    for name, payload in {
        "preflight.json": {"apple_silicon": True},
        "quality_eval_short_summary.json": {"status": "PASS", "model": "demo"},
        "cert_manifest.json": {"result": "PASS"},
        "bit_budget_sweep_summary.json": {"metric_family": "bit_budget_sweep"},
        "inner_product_bias_summary.json": {"metric_family": "inner_product_bias"},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    records = collect_run_artifacts(tmp_path)

    assert len(records) == 1
    assert records[0]["run_id"] == "demo-run"
