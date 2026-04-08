import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_import_surface() -> None:
    """Internal code and docs must prefer turboquant.integrations.mlx."""
    scan_dirs = [
        "integrations",
        "turboquant",
        "mlx_lm",
        "tests",
        "docs",
        "scripts",
        "benchmarks",
    ]
    forbidden_patterns = (
        re.compile(r"from integrations\.mlx"),
        re.compile(r"import integrations\.mlx"),
    )
    allowed_files = {
        "integrations/mlx/upgrade.py",
        "tests/unit_static/test_canonical_import_surface.py",
    }
    found_violations: list[str] = []

    for scan_dir in scan_dirs:
        abs_scan_dir = REPO_ROOT / scan_dir
        if not abs_scan_dir.exists():
            continue

        for full_path in abs_scan_dir.rglob("*"):
            if full_path.suffix not in {".py", ".md"}:
                continue

            rel_path = full_path.relative_to(REPO_ROOT).as_posix()
            if rel_path in allowed_files:
                continue

            try:
                content = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for pattern in forbidden_patterns:
                if pattern.search(content):
                    found_violations.append(f"{rel_path}: matches {pattern.pattern}")

    if found_violations:
        pytest.fail(
            "Found canonical import violations:\n" + "\n".join(found_violations)
        )
