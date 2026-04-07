from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from benchmarks.runtime_cert.research_report_schema import missing_common_fields

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(script_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script_path), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_common_research_fields(payload: dict[str, Any]) -> None:
    missing = missing_common_fields(payload)
    assert not missing, f"missing common research fields: {missing}"
