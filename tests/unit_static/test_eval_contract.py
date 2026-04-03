"""
tests/unit_static/test_eval_contract.py — eval/cert caller contract tests.

Verifies that every helper that calls ``upgrade_cache_list`` has a
``model_family`` parameter in its signature.  These are static AST/signature
checks only — no MLX execution required.

Contract coverage
-----------------
1. ``perplexity_report_has_model_family`` — ``turboquant.eval.perplexity.perplexity_report``
   must accept a ``model_family`` keyword argument.

2. ``memory_report_has_model_family`` — ``turboquant.eval.memory.memory_report``
   must accept a ``model_family`` keyword argument.

3. ``eval_one_prompt_has_model_family`` — the ``_eval_one_prompt`` helper in
   ``benchmarks/runtime_cert/run_quality_eval.py`` must accept a
   ``model_family`` keyword argument so that it does not silently bypass the
   fail-closed support gate.

4. ``perplexity_report_model_family_default`` — ``model_family`` defaults to
   ``"llama"`` (a supported family) in ``perplexity_report``.

5. ``memory_report_model_family_default`` — ``model_family`` defaults to
   ``"llama"`` (a supported family) in ``memory_report``.

6. ``infer_model_family_checks_model_type_attr`` — ``_infer_model_family`` in
   ``mlx_lm/generate.py`` resolves a ``model.model_type`` attribute before
   falling back to class-name search.  Guards against regression where
   ``TinyModel.model_type="llama"`` stops being honoured.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_repo_to_path() -> bool:
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        return True
    return False


# ---------------------------------------------------------------------------
# 1. perplexity_report has model_family param
# ---------------------------------------------------------------------------


def test_perplexity_report_has_model_family() -> None:
    """perplexity_report must accept a model_family keyword argument."""
    injected = _add_repo_to_path()
    try:
        from turboquant.eval.perplexity import perplexity_report

        sig = inspect.signature(perplexity_report)
        assert "model_family" in sig.parameters, (
            "turboquant.eval.perplexity.perplexity_report is missing the "
            "'model_family' parameter.  Without it, upgrade_cache_list is "
            "called without a family, which raises UnsupportedModelError at "
            "runtime (fail-closed gate)."
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 2. memory_report has model_family param
# ---------------------------------------------------------------------------


def test_memory_report_has_model_family() -> None:
    """memory_report must accept a model_family keyword argument."""
    injected = _add_repo_to_path()
    try:
        from turboquant.eval.memory import memory_report

        sig = inspect.signature(memory_report)
        assert "model_family" in sig.parameters, (
            "turboquant.eval.memory.memory_report is missing the "
            "'model_family' parameter.  Without it, upgrade_cache_list is "
            "called without a family, which raises UnsupportedModelError at "
            "runtime (fail-closed gate)."
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 3. _eval_one_prompt has model_family param (AST check — not importable)
# ---------------------------------------------------------------------------


def test_eval_one_prompt_has_model_family() -> None:
    """_eval_one_prompt in run_quality_eval.py must accept model_family."""
    script_path = REPO_ROOT / "benchmarks" / "runtime_cert" / "run_quality_eval.py"
    assert script_path.exists(), (
        "benchmarks/runtime_cert/run_quality_eval.py not found"
    )
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    fn_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_eval_one_prompt":
            fn_node = node
            break

    assert fn_node is not None, (
        "_eval_one_prompt function not found in run_quality_eval.py"
    )

    param_names = [arg.arg for arg in fn_node.args.args]
    assert "model_family" in param_names, (
        "_eval_one_prompt in run_quality_eval.py is missing the 'model_family' "
        "parameter.  Without it, upgrade_cache_list is called without a family "
        "and raises UnsupportedModelError at certification time."
    )


# ---------------------------------------------------------------------------
# 4. perplexity_report model_family defaults to "llama"
# ---------------------------------------------------------------------------


def test_perplexity_report_model_family_default() -> None:
    """perplexity_report model_family must default to 'llama' (a supported family)."""
    injected = _add_repo_to_path()
    try:
        from turboquant.eval.perplexity import perplexity_report

        sig = inspect.signature(perplexity_report)
        param = sig.parameters.get("model_family")
        assert param is not None, "model_family parameter not found"
        assert param.default == "llama", (
            f"perplexity_report model_family default is {param.default!r}; "
            "expected 'llama'.  The default must be a member of SUPPORTED_FAMILIES."
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 5. memory_report model_family defaults to "llama"
# ---------------------------------------------------------------------------


def test_memory_report_model_family_default() -> None:
    """memory_report model_family must default to 'llama' (a supported family)."""
    injected = _add_repo_to_path()
    try:
        from turboquant.eval.memory import memory_report

        sig = inspect.signature(memory_report)
        param = sig.parameters.get("model_family")
        assert param is not None, "model_family parameter not found"
        assert param.default == "llama", (
            f"memory_report model_family default is {param.default!r}; "
            "expected 'llama'.  The default must be a member of SUPPORTED_FAMILIES."
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 6. _infer_model_family checks model_type attribute (AST check)
# ---------------------------------------------------------------------------


def test_infer_model_family_checks_model_type_attr() -> None:
    """_infer_model_family in generate.py must check model.model_type before class search."""
    generate_py = REPO_ROOT / "mlx_lm" / "generate.py"
    assert generate_py.exists(), "mlx_lm/generate.py not found"

    tree = ast.parse(generate_py.read_text(encoding="utf-8"))

    fn_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_infer_model_family"
        ):
            fn_node = node
            break

    assert fn_node is not None, "_infer_model_family not found in generate.py"

    # Collect string constants inside the function body — "model_type" must appear.
    source_lines = generate_py.read_text(encoding="utf-8").splitlines()
    start = fn_node.lineno - 1
    end = fn_node.end_lineno  # type: ignore[attr-defined]
    fn_source = "\n".join(source_lines[start:end])

    assert "model_type" in fn_source, (
        "_infer_model_family does not reference 'model_type'.  "
        "It must check model.model_type (standard mlx-lm attribute) before "
        "falling back to class-name search, so that TinyModel and real models "
        "with an explicit model_type are resolved correctly."
    )
