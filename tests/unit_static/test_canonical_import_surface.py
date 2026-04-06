import os
import re

import pytest


def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_canonical_import_surface():
    """Ensure all internal code and docs use the turboquant.integrations.mlx namespace."""
    root = get_repo_root()

    # Directories to scan
    scan_dirs = ["turboquant", "mlx_lm", "tests", "docs", "scripts", "benchmarks"]

    # Patterns that should no longer be used for primary logic
    forbidden_patterns = [
        re.compile(r"from integrations\.mlx"),
        re.compile(r"import integrations\.mlx"),
    ]

    # Exceptions: The shims themselves are allowed to have these imports
    # and any compatibility tests explicitly checking the shims.
    allowed_files = [
        "integrations/mlx/cache_adapter.py",
        "integrations/mlx/upgrade.py",
        "test_canonical_import_surface.py",  # self
    ]

    found_violations = []

    for sdir in scan_dirs:
        abs_sdir = os.path.join(root, sdir)
        if not os.path.exists(abs_sdir):
            continue

        for r, d, files in os.walk(abs_sdir):
            for f in files:
                if not f.endswith((".py", ".md")):
                    continue

                full_path = os.path.join(r, f)
                rel_path = os.path.relpath(full_path, root)

                if any(rel_path == allowed for allowed in allowed_files):
                    continue

                with open(full_path, encoding="utf-8") as f_handle:
                    try:
                        content = f_handle.read()
                        for pattern in forbidden_patterns:
                            if pattern.search(content):
                                found_violations.append(
                                    f"{rel_path}: matches {pattern.pattern}"
                                )
                    except UnicodeDecodeError:
                        continue

    if found_violations:
        pytest.fail(
            "Found canonical import violations:\n" + "\n".join(found_violations)
        )
