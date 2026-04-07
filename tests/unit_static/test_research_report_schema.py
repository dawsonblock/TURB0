from __future__ import annotations

from benchmarks.runtime_cert.research_report_schema import (
    COMMON_RESEARCH_REPORT_FIELDS,
    build_artifact_paths,
    build_research_report,
    missing_common_fields,
)


def test_build_research_report_exposes_common_fields() -> None:
    payload = build_research_report(
        schema_version="1",
        metric_family="demo",
        run_id="run-1",
        environment={"timestamp": "2026-04-07T00:00:00+00:00"},
        preset="paper_mse",
        family="llama",
        scope="research-only",
        mode="demo_mode",
        status="ok",
        metrics={"demo_metric": 1.0},
        artifact_paths=build_artifact_paths(summary_json="demo.json"),
        notes=["demo note"],
        extra_field=True,
    )

    assert missing_common_fields(payload) == []
    assert tuple(field for field in COMMON_RESEARCH_REPORT_FIELDS if field in payload) == COMMON_RESEARCH_REPORT_FIELDS
    assert payload["extra_field"] is True
