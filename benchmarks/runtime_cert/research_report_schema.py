from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

COMMON_RESEARCH_REPORT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "metric_family",
    "run_id",
    "timestamp",
    "preset",
    "family",
    "scope",
    "mode",
    "status",
    "metrics",
    "artifact_paths",
    "notes",
)


def _slugify(value: str, *, limit: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug[:limit] or "research"


def build_run_id(*, timestamp: str, label: str, mode: str) -> str:
    stamp = (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    return f"{stamp}_{_slugify(label)}_{_slugify(mode, limit=24)}"


def build_artifact_paths(**paths: str | Path) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def build_research_report(
    *,
    schema_version: str,
    metric_family: str,
    run_id: str,
    environment: Mapping[str, Any],
    preset: str,
    family: str,
    scope: str,
    mode: str,
    status: str,
    metrics: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    notes: list[str],
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "metric_family": metric_family,
        "run_id": run_id,
        "timestamp": str(environment.get("timestamp", "")),
        "preset": preset,
        "family": family,
        "scope": scope,
        "mode": mode,
        "status": status,
        "metrics": dict(metrics),
        "artifact_paths": dict(artifact_paths),
        "notes": list(notes),
    }
    payload.update(extra)
    return payload


def missing_common_fields(payload: Mapping[str, Any]) -> list[str]:
    return [field for field in COMMON_RESEARCH_REPORT_FIELDS if field not in payload]
