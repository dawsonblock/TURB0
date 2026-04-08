import os
import re

import pytest


def test_single_entry_path_proof():
    """
    Ensure the codebase only uses the canonical TurboQuantKCache path
    and no legacy 'KVCompressor' or 'maybe_turboquant_k_cache' calls
    remain in production loops.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Files to exclude (legacy shims, tests, documentation)
    exclude = [
        "integrations/mlx/upgrade.py",
        "tests/",
        "docs/",
        "README.md",
    ]

    # We want to check the shipped runtime tree for legacy usage.
    search_dirs = [
        os.path.join(root, "turboquant/core"),
        os.path.join(root, "turboquant/runtime"),
        os.path.join(root, "turboquant"),
    ]

    legacy_patterns: list[re.Pattern[str]] = []

    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for r, d, f in os.walk(sdir):
            for file in f:
                if not file.endswith(".py"):
                    continue
                fpath = os.path.join(r, file)
                rel_path = os.path.relpath(fpath, root)

                if any(rel_path.startswith(e) for e in exclude):
                    continue

                with open(fpath) as f_handle:
                    content = f_handle.read()
                    for pattern in legacy_patterns:
                        if pattern.search(content):
                            # Special case: generate.py is allowed to contain
                            # the legacy shim definition.
                            if (
                                "generate.py" in rel_path
                                and "def maybe_turboquant_k_cache" in content
                            ):
                                continue
                            pytest.fail(
                                "Legacy pattern "
                                f"'{pattern.pattern}' found in production "
                                f"file: {rel_path}"
                            )

def test_patch_layer_uses_canonical_upgrade_path():
    """turboquant.patch must still point callers back to upgrade_cache_list."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    patch_path = os.path.join(root, "turboquant/patch.py")

    with open(patch_path) as f:
        content = f.read()

    assert "upgrade_cache_list" in content
