"""
tests/unit_static/test_governance.py — structural governance contracts.

These tests verify invariants that must hold WITHOUT executing any MLX code.
They are the static enforcement layer for the repo's stated architectural
claims.  Every test here should pass on any machine (Linux CI, macOS x86,
developer laptops) regardless of whether MLX or Apple Silicon is present.

Contract coverage
-----------------
1. ``no_mlx_import_in_unit_static`` — none of the files in this directory
   import ``mlx``; if they did, they would not be portable and would
   undermine the static-test contract.

2. ``noxfile_excludes_unit_from_mlx_session`` — the noxfile's ``tests_mlx``
    session must NOT include the dead ``tests/unit/`` path now that the
    maintained MLX test layout is ``tests/unit_mlx/`` plus
    ``tests/integration_mlx/``.

3. ``support_module_has_expected_families`` — ``SUPPORTED_FAMILIES`` in
    ``turboquant/runtime/support.py`` must match the allowlisted families in
    ``turboquant/contract.json``. Any addition to this set must
    be deliberate and come with runtime-cert coverage; if it silently changes,
    this test will catch it.

4. ``unsupported_family_raises_unsupported_model_error`` — calling
   ``assert_supported_model_family`` with an unlisted family must raise
   ``UnsupportedModelError``.  This verifies Gate 2 is wired correctly.

5. ``v_enabled_default_matches_architecture_doc`` — ``turboquant/config.py``
   defaults ``v_enabled=True``; the architecture doc must say "enabled by
   default".  Regression guard against the v_enabled contradiction fixed in
   Phase 4.

6. ``upgrade_cache_list_none_family_raises`` — ``upgrade_cache_list`` must
   raise ``UnsupportedModelError`` when called with ``model_family=None``.
   Ensures the support gate is truly fail-closed (not fail-open).

7. ``infer_model_family_returns_supported_or_none`` — ``_infer_model_family``
   must only return values from ``SUPPORTED_FAMILIES`` or ``None``.  Prevents
   the inference list from drifting ahead of the allowlist.

8. ``upgrade_cache_list_none_docstring_correct`` — the ``upgrade_cache_list``
   docstring must NOT claim ``None`` is a valid bypass for exploratory code.

9. ``architecture_doc_no_online_softmax_claim`` — ``docs/architecture.md``
   must not claim "online softmax" or "2-accumulator" because the actual
   implementation materialises all scores then calls a single ``mx.softmax``.

10. ``architecture_doc_no_rotate_queries_for_attention`` — the architecture
    doc must not claim ``rotate_queries_for_attention()`` is called at the
    top level; rotation happens inside ``score_block()``.

11. ``kvcompressor_is_marked_as_alias`` — ``turboquant/__init__.py`` must
    explicitly document ``KVCompressor`` as a compatibility alias so callers
    understand it is not the primary API.

12. ``attention_hot_path_does_not_read_block_tokens`` —
    ``turboquant/runtime/attention.py`` must not start reading ``block_tokens``
    unless the public contract is deliberately widened.

13. ``only_experimental_metal_shader_path_exists`` — the repo must not retain
    a second non-canonical ``turboquant/kernels/metal/decode_k.metal`` copy;
    the experimental namespace is the only supported shader location.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 1. No mlx import in this directory
# ---------------------------------------------------------------------------


def test_no_mlx_import_in_unit_static() -> None:
    """No file in tests/unit_static/ may import mlx."""
    this_dir = Path(__file__).parent
    violations: list[str] = []
    for py in sorted(this_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "mlx" for alias in node.names):
                    violations.append(py.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "mlx":
                    violations.append(py.name)
    assert not violations, (
        "These files in tests/unit_static/ import mlx (forbidden):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# 2. noxfile excludes the dead tests/unit/ path from tests_mlx
# ---------------------------------------------------------------------------


def test_noxfile_excludes_unit_from_mlx_session() -> None:
    """tests/unit/ must stay out of the tests_mlx nox session."""
    noxfile = REPO_ROOT / "noxfile.py"
    assert noxfile.exists(), "noxfile.py not found at repo root"

    text = noxfile.read_text(encoding="utf-8")
    tree = ast.parse(text)

    # Walk the AST to find the tests_mlx function definition.
    mlx_session_src: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "tests_mlx":
            # Grab the source lines for just this function.
            lines = text.splitlines()
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            mlx_session_src = "\n".join(lines[start:end])
            break

    assert mlx_session_src is not None, (
        "Could not find 'tests_mlx' function in noxfile.py"
    )
    assert "tests/unit/" not in mlx_session_src, (
        "noxfile.py tests_mlx session still references 'tests/unit/'.\n"
        "The maintained MLX layout is tests/unit_mlx/ plus "
        "tests/integration_mlx/."
    )


def test_compatibility_conftest_tracks_current_test_layout() -> None:
    """turboquant/tests/conftest.py must not point at dead test paths."""
    conftest_path = REPO_ROOT / "turboquant" / "tests" / "conftest.py"
    assert conftest_path.exists(), "turboquant/tests/conftest.py not found"

    text = conftest_path.read_text(encoding="utf-8")
    assert "tests/unit/" not in text, (
        "turboquant/tests/conftest.py still points at the dead tests/unit/ path."
    )
    assert "tests/unit_static/" in text
    assert "tests/unit_mlx/" in text
    assert "tests/integration_mlx/" in text


# ---------------------------------------------------------------------------
# 3. SUPPORTED_FAMILIES content
# ---------------------------------------------------------------------------


def test_support_module_has_expected_families() -> None:
    """SUPPORTED_FAMILIES must match the allowlisted contract families."""
    contract_json = REPO_ROOT / "turboquant" / "contract.json"
    assert contract_json.exists(), "turboquant/contract.json not found"

    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    expected = {
        family["name"]
        for family in contract["families"]
        if family["status"] == "allowlisted"
    }
    assert expected == {"llama", "gemma"}, (
        "The machine-readable contract should currently allowlist only llama and gemma."
    )

    repo_str = str(REPO_ROOT)
    injected = repo_str not in sys.path
    if injected:
        sys.path.insert(0, repo_str)
    try:
        from turboquant.runtime.support import SUPPORTED_FAMILIES

        assert set(SUPPORTED_FAMILIES) == expected, (
            f"SUPPORTED_FAMILIES mismatch.\n"
            f"  Expected : {sorted(expected)}\n"
            f"  Got      : {sorted(SUPPORTED_FAMILIES)}\n"
            "If you are adding a new family, update the contract JSON and "
            "runtime-cert coverage together."
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable in this env: {exc}")
    finally:
        if injected:
            sys.path.remove(repo_str)


# ---------------------------------------------------------------------------
# 4. Gate 2: unsupported family raises UnsupportedModelError
# ---------------------------------------------------------------------------


def test_unsupported_family_raises_unsupported_model_error() -> None:
    """assert_supported_model_family('mixtral') must raise the typed error."""
    # Patch sys.path so we can import without installing the package.
    repo_str = str(REPO_ROOT)
    injected = repo_str not in sys.path
    if injected:
        sys.path.insert(0, repo_str)
    try:
        from turboquant.errors import UnsupportedModelError
        from turboquant.runtime.support import assert_supported_model_family

        with pytest.raises(UnsupportedModelError):
            assert_supported_model_family("mixtral")

        # Supported families must NOT raise.
        assert_supported_model_family("llama")
        assert_supported_model_family("gemma")
        assert_supported_model_family("llama3_1")  # normalisation check
        assert_supported_model_family("Gemma2")  # case-insensitive

    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable in this env: {exc}")
    finally:
        if injected:
            sys.path.remove(repo_str)


# ---------------------------------------------------------------------------
# 5. v_enabled default matches architecture.md
# ---------------------------------------------------------------------------


def test_v_enabled_default_matches_architecture_doc() -> None:
    """architecture.md must state v_enabled is enabled by default."""
    arch_doc = REPO_ROOT / "docs" / "architecture.md"
    assert arch_doc.exists(), "docs/architecture.md not found"

    text = arch_doc.read_text(encoding="utf-8")

    # The fixed text says "enabled by default" — the old (wrong) text said
    # "disabled by default for some model families".
    assert "disabled by default for some model families" not in text, (
        "docs/architecture.md still claims V quantisation is disabled by "
        "default for some model families, which contradicts config.py "
        "(v_enabled=True).  Fix the documentation."
    )
    # Positive assertion: the corrected phrasing must be present.
    assert "enabled by default" in text, (
        "docs/architecture.md must explicitly state that v_enabled is "
        "enabled by default.  Check Phase 4 of the cleanup notes."
    )


# ---------------------------------------------------------------------------
# 6. upgrade_cache_list(model_family=None) must raise, not silently bypass gate
# ---------------------------------------------------------------------------


def test_upgrade_cache_list_none_family_raises() -> None:
    """upgrade_cache_list must raise when model_family is None."""
    repo_str = str(REPO_ROOT)
    injected = repo_str not in sys.path
    if injected:
        sys.path.insert(0, repo_str)
    try:
        from turboquant.config import TurboQuantConfig
        from turboquant.errors import UnsupportedModelError
        from turboquant.integrations.mlx.upgrade import upgrade_cache_list

        cfg = TurboQuantConfig()
        with pytest.raises(UnsupportedModelError):
            upgrade_cache_list([], k_start=0, config=cfg, model_family=None)
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable in this env: {exc}")
    finally:
        if injected:
            sys.path.remove(repo_str)


# ---------------------------------------------------------------------------
# 7. _infer_model_family only returns supported families or None
# ---------------------------------------------------------------------------


def test_infer_model_family_returns_supported_or_none() -> None:
    """_infer_model_family must return a SUPPORTED_FAMILIES member or None.

    Catches drift where the inference list grows ahead of the allowlist,
    which would create a false sense of coverage for unsupported families.
    """
    repo_str = str(REPO_ROOT)
    injected = repo_str not in sys.path
    if injected:
        sys.path.insert(0, repo_str)
    try:
        from turboquant.runtime.support import SUPPORTED_FAMILIES

        # Read generate.py and extract the for-loop families list from
        # _infer_model_family without executing it.
        gen_py = REPO_ROOT / "mlx_lm" / "generate.py"
        text = gen_py.read_text(encoding="utf-8")
        tree = ast.parse(text)

        # Find the _infer_model_family function and look for string constants
        # used in a for-loop (the family membership list).
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_infer_model_family":
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        # Inspect the iterator for string constants.
                        if isinstance(
                            child.iter,
                            (ast.Tuple, ast.List, ast.Set),
                        ):
                            for elt in child.iter.elts:
                                if isinstance(
                                    elt,
                                    ast.Constant,
                                ) and isinstance(elt.value, str):
                                    assert elt.value in SUPPORTED_FAMILIES, (
                                        f"_infer_model_family iterates over "
                                        f"'{elt.value}' but that family is "
                                        "not in "
                                        f"SUPPORTED_FAMILIES "
                                        f"({sorted(SUPPORTED_FAMILIES)}).  "
                                        "Either add it to the allowlist with "
                                        "runtime-cert coverage or "
                                        "remove it from the inference loop."
                                    )
                        break  # only check the first for-loop in the function
                break
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable in this env: {exc}")
    finally:
        if injected:
            sys.path.remove(repo_str)


# ---------------------------------------------------------------------------
# 8. upgrade_cache_list docstring must not claim None is a valid bypass
# ---------------------------------------------------------------------------


def test_upgrade_cache_list_none_docstring_correct() -> None:
    """upgrade_cache_list docstring must not claim None bypasses the gate."""
    upgrade_py = REPO_ROOT / "turboquant" / "integrations" / "mlx" / "upgrade.py"
    assert upgrade_py.exists(), "turboquant/integrations/mlx/upgrade.py not found"

    text = upgrade_py.read_text(encoding="utf-8")
    assert "intentionally bypass the allowlist" not in text, (
        "upgrade_cache_list docstring still claims passing model_family=None "
        "is a valid way to bypass the allowlist from 'exploratory code "
        "paths'. "
        "That is incorrect: None now raises UnsupportedModelError.  "
        "Update the docstring."
    )


# ---------------------------------------------------------------------------
# 9. architecture.md must not claim online softmax
# ---------------------------------------------------------------------------


def test_architecture_doc_no_online_softmax_claim() -> None:
    """architecture.md must not claim online-softmax-style behavior."""
    arch_doc = REPO_ROOT / "docs" / "architecture.md"
    assert arch_doc.exists(), "docs/architecture.md not found"

    text = arch_doc.read_text(encoding="utf-8")
    for forbidden in ("online softmax", "2-accumulator", "log-sum-exp lse"):
        assert forbidden not in text, (
            f"docs/architecture.md still contains '{forbidden}', implying an "
            "online-softmax implementation.  The actual code concatenates all "
            "block scores then calls mx.softmax once.  Fix the documentation."
        )


# ---------------------------------------------------------------------------
# 10. architecture.md must not claim rotate_queries_for_attention at top level
# ---------------------------------------------------------------------------


def test_architecture_doc_no_rotate_queries_for_attention() -> None:
    """architecture.md must not claim rotate_queries_for_attention()."""
    arch_doc = REPO_ROOT / "docs" / "architecture.md"
    assert arch_doc.exists(), "docs/architecture.md not found"

    text = arch_doc.read_text(encoding="utf-8")
    assert "rotate_queries_for_attention" not in text, (
        "docs/architecture.md still references "
        "'rotate_queries_for_attention()'. That function does not exist; "
        "query rotation happens inside score_block() "
        "via FixedRotation.apply().  Fix the documentation."
    )


# ---------------------------------------------------------------------------
# 11. KVCompressor must be documented as compatibility alias in __init__.py
# ---------------------------------------------------------------------------


def test_kvcompressor_is_marked_as_alias() -> None:
    """turboquant/__init__.py must document KVCompressor as an alias."""
    init_py = REPO_ROOT / "turboquant" / "__init__.py"
    assert init_py.exists(), "turboquant/__init__.py not found"

    text = init_py.read_text(encoding="utf-8")
    # Both the module docstring line and the __all__ comment must say "alias".
    assert text.lower().count("alias") >= 2, (
        "turboquant/__init__.py must mark KVCompressor as a compatibility "
        "alias in at least two places (module docstring and __all__ "
        "comment) so callers know to prefer TurboQuantKVCache.  "
        "Currently found fewer than 2 occurrences "
        "of the word 'alias'."
    )


# ---------------------------------------------------------------------------
# 12. attention hot path must not read block_tokens
# ---------------------------------------------------------------------------


def test_attention_hot_path_does_not_read_block_tokens() -> None:
    """turboquant/runtime/attention.py must not read block_tokens."""
    attention_py = REPO_ROOT / "turboquant" / "runtime" / "attention.py"
    assert attention_py.exists(), "turboquant/runtime/attention.py not found"

    text = attention_py.read_text(encoding="utf-8")
    assert "block_tokens" not in text, (
        "turboquant/runtime/attention.py now references block_tokens.  If "
        "that is intentional, widen the documented contract and update the "
        "static checks at "
        "the same time."
    )


# ---------------------------------------------------------------------------
# 13. Only the experimental Metal shader path may exist
# ---------------------------------------------------------------------------


def test_only_experimental_metal_shader_path_exists() -> None:
    """Non-canonical duplicate Metal shader paths must stay absent."""
    canonical = (
        REPO_ROOT
        / "turboquant"
        / "experimental"
        / "kernels"
        / "metal"
        / "decode_k.metal"
    )
    duplicate = REPO_ROOT / "turboquant" / "kernels" / "metal" / "decode_k.metal"

    assert canonical.exists(), (
        "Canonical experimental Metal shader asset missing at "
        "turboquant/experimental/kernels/metal/decode_k.metal."
    )
    assert not duplicate.exists(), (
        "Found stale duplicate Metal shader asset at "
        "turboquant/kernels/metal/decode_k.metal. Keep the experimental "
        "namespace as the single canonical shader location."
    )
