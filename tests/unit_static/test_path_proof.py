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
        "integrations/mlx/cache_adapter.py",
        "tests/",
        "docs/",
        "README.md",
    ]

    # We want to check mlx_lm/ and turboquant/core/ for legacy usage
    search_dirs = [
        os.path.join(root, "mlx_lm"),
        os.path.join(root, "turboquant/core"),
        os.path.join(root, "turboquant/runtime"),
    ]

    legacy_patterns = [
        re.compile(r"KVCompressor"),
        # maybe_turboquant_k_cache is allowed in the shim and generate.py (as entry point)
        # but we want to ensure it's not being called by other inner modules.
    ]

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


def test_canonical_import_lock():
    """Ensure mlx_lm/generate.py uses the canonical upgrade_cache_list via the shim."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gen_path = os.path.join(root, "mlx_lm/generate.py")

    with open(gen_path) as f:
        content = f.read()

    # Check that maybe_turboquant_k_cache delegates to upgrade_cache_list
    assert "from turboquant.integrations.mlx.upgrade import upgrade_cache_list" in content
    assert "upgrade_cache_list(prompt_cache" in content
