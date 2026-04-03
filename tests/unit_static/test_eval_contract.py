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


# ---------------------------------------------------------------------------
# 7. _collect_logits_compressed docstring documents the gate bypass
# ---------------------------------------------------------------------------


def test_compare_collect_logits_compressed_documents_bypass() -> None:
    """_collect_logits_compressed must document that it bypasses the support gate."""
    compare_py = REPO_ROOT / "turboquant" / "eval" / "compare.py"
    assert compare_py.exists(), "turboquant/eval/compare.py not found"

    tree = ast.parse(compare_py.read_text(encoding="utf-8"))

    fn_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_collect_logits_compressed"
        ):
            fn_node = node
            break

    assert fn_node is not None, (
        "_collect_logits_compressed not found in turboquant/eval/compare.py"
    )

    # The first statement in the function body must be a docstring.
    docstring = ast.get_docstring(fn_node) or ""
    assert "bypass" in docstring.lower(), (
        "_collect_logits_compressed must document in its docstring that it "
        "bypasses the model-family support gate (i.e. the word 'bypass' must "
        "appear).  Without this warning, contributors may copy the pattern "
        "thinking it is a valid production path."
    )


# ---------------------------------------------------------------------------
# 8. KVCache.to_turboquant docstring documents the gate bypass
# ---------------------------------------------------------------------------


def test_to_turboquant_documents_gate_bypass() -> None:
    """KVCache.to_turboquant must document that it bypasses the support gate."""
    cache_py = REPO_ROOT / "mlx_lm" / "models" / "cache.py"
    assert cache_py.exists(), "mlx_lm/models/cache.py not found"

    tree = ast.parse(cache_py.read_text(encoding="utf-8"))

    fn_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "to_turboquant":
            fn_node = node
            break

    assert fn_node is not None, (
        "to_turboquant not found in mlx_lm/models/cache.py"
    )

    docstring = ast.get_docstring(fn_node) or ""
    assert "bypass" in docstring.lower(), (
        "KVCache.to_turboquant must document in its docstring that it bypasses "
        "the model-family support gate (i.e. the word 'bypass' must appear).  "
        "Without this warning callers may not realise they are skipping the "
        "fail-closed gate enforced by upgrade_cache_list."
    )
    assert "deprecated" in docstring.lower(), (
        "KVCache.to_turboquant must say it is deprecated so contributors do not "
        "treat it as a peer runtime surface."
    )
    assert "compatibility" in docstring.lower(), (
        "KVCache.to_turboquant must identify itself as a compatibility helper."
    )


# ---------------------------------------------------------------------------
# 9. upgrade_cache_list rejects model_family=None (fail-closed gate)
# ---------------------------------------------------------------------------


def test_upgrade_cache_list_rejects_none_model_family() -> None:
    """upgrade_cache_list must raise UnsupportedModelError when model_family=None."""
    injected = _add_repo_to_path()
    try:
        from turboquant.errors import UnsupportedModelError
        from turboquant.integrations.mlx.upgrade import upgrade_cache_list

        # A minimal stub — the gate check fires before any cache access.
        class _StubCache:
            offset = 0

        with pytest.raises(UnsupportedModelError):
            upgrade_cache_list(
                [_StubCache()],
                k_start=0,
                config=object(),
                model_family=None,
            )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 10. EventLog records and flushes in memory (no MLX required)
# ---------------------------------------------------------------------------


def test_event_log_records_and_flushes_in_memory() -> None:
    """EventLog.record() + flush() must work with artifact_dir=None (memory-only).

    This proves the persistence-layer infrastructure compiles and is usable
    independently of the canonical runtime path.
    """
    injected = _add_repo_to_path()
    try:
        from turboquant.runtime.events import CacheUpgradeEvent, EventLog

        log = EventLog(artifact_dir=None)

        evt = CacheUpgradeEvent(
            layer_index=0,
            token_index=64,
            old_type="KVCache",
            new_type="TurboQuantKCache",
            old_bytes=131072,
            new_bytes=16384,
        )
        log.record(evt)

        assert len(log.events) == 1, "EventLog.record() must append the event"
        assert log.events[0].layer_index == 0

        # flush() with artifact_dir=None must be a no-op (not raise).
        log.flush()

        # After flush the event is still accessible.
        assert len(log.events) == 1, (
            "EventLog.events must still return all recorded events after flush()"
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


def test_record_runtime_upgrade_events_converts_runtime_events() -> None:
    """record_runtime_upgrade_events must bridge runtime events into EventLog explicitly."""
    injected = _add_repo_to_path()
    try:
        from turboquant.runtime.events import EventLog, record_runtime_upgrade_events

        class _RuntimeUpgradeEvent:
            def __init__(
                self,
                *,
                upgraded: bool,
                layer_index: int,
                offset_at_upgrade: int,
                old_type: str,
                new_type: str,
                old_bytes: int,
                new_bytes: int,
            ) -> None:
                self.upgraded = upgraded
                self.layer_index = layer_index
                self.offset_at_upgrade = offset_at_upgrade
                self.old_type = old_type
                self.new_type = new_type
                self.old_bytes = old_bytes
                self.new_bytes = new_bytes

        log = EventLog(artifact_dir=None)
        recorded = record_runtime_upgrade_events(
            log,
            [
                _RuntimeUpgradeEvent(
                    upgraded=True,
                    layer_index=2,
                    offset_at_upgrade=128,
                    old_type="KVCache",
                    new_type="TurboQuantKCache",
                    old_bytes=262144,
                    new_bytes=65536,
                ),
                _RuntimeUpgradeEvent(
                    upgraded=False,
                    layer_index=3,
                    offset_at_upgrade=128,
                    old_type="KVCache",
                    new_type="KVCache",
                    old_bytes=0,
                    new_bytes=0,
                ),
            ],
        )

        assert recorded == 1, "Only successful runtime upgrades should be persisted."
        assert len(log.events) == 1, "Exactly one persistence event should be recorded."
        persisted = log.events[0]
        assert persisted.layer_index == 2
        assert persisted.token_index == 128
        assert persisted.old_type == "KVCache"
        assert persisted.new_type == "TurboQuantKCache"
    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable: {exc}")
    finally:
        if injected:
            sys.path.remove(str(REPO_ROOT))


def test_runtime_events_module_marks_persistence_optional() -> None:
    """runtime/events.py must describe itself as the optional persistence layer."""
    events_py = REPO_ROOT / "turboquant" / "runtime" / "events.py"
    assert events_py.exists(), "turboquant/runtime/events.py not found"

    text = events_py.read_text(encoding="utf-8")
    lowered = text.lower()
    assert 'does **not** automatically' in text or 'does not automatically' in lowered, (
        "runtime/events.py must state that canonical runtime execution does not automatically persist events."
    )
    assert 'optional persistence' in lowered, (
        "runtime/events.py must describe itself as the optional persistence-side layer."
    )
    assert 'record_runtime_upgrade_events' in text, (
        "runtime/events.py must expose the explicit runtime-to-persistence adapter."
    )


def test_upgrade_module_marks_runtime_events_as_non_persistent() -> None:
    """upgrade.py must describe its returned events as lightweight runtime results."""
    upgrade_py = REPO_ROOT / "turboquant" / "integrations" / "mlx" / "upgrade.py"
    assert upgrade_py.exists(), "turboquant/integrations/mlx/upgrade.py not found"

    text = upgrade_py.read_text(encoding="utf-8")
    lowered = text.lower()
    assert 'lightweight' in lowered and 'does not automatically persist' in lowered, (
        "upgrade.py must state that its returned CacheUpgradeEvent objects are lightweight runtime results that are not automatically persisted."
    )
    assert 'record_runtime_upgrade_events' in text, (
        "upgrade.py must point persistence-oriented callers at record_runtime_upgrade_events(...)."
    )


def test_benchmark_script_uses_explicit_event_adapter() -> None:
    """scripts/benchmark.py must opt into persistence explicitly via the event adapter."""
    script_path = REPO_ROOT / "scripts" / "benchmark.py"
    assert script_path.exists(), "scripts/benchmark.py not found"

    text = script_path.read_text(encoding="utf-8")
    assert 'record_runtime_upgrade_events' in text, (
        "scripts/benchmark.py must use record_runtime_upgrade_events(...) instead of implying automatic event persistence."
    )
    assert 'EventLog' in text, (
        "scripts/benchmark.py must construct an EventLog explicitly when it wants persisted events."
    )
    assert 'tracker.write(event_log=event_log)' in text, (
        "scripts/benchmark.py must pass an explicit event_log to tracker.write()."
    )
