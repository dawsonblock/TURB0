"""Integration tests for _infer_model_family in generate.py.

Verifies the auto-detection logic using mock model classes — no real model
loading required.

Requires MLX (Apple Silicon) for the import chain.

Behavioural contract (post Phase-1 cleanup)
-------------------------------------------
``_infer_model_family`` only returns a family string when that family is in
``SUPPORTED_FAMILIES`` (currently ``{"llama", "gemma"}``).  Any other family
— including formerly-detected ones such as "mistral", "phi", "qwen" — now
returns ``None`` so the caller skips the TurboQuant upgrade rather than
bypassing the support gate.
"""

from __future__ import annotations

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

pytest.importorskip("mlx.core")

from mlx_lm.generate import _infer_model_family  # noqa: E402

# ── Mock model classes living in known module-path families ───────────────────


class _FakeLlamaModel:
    """Lives in a path containing 'llama'."""


_FakeLlamaModel.__module__ = "mlx_lm.models.llama"


class _FakeGemmaModel:
    """Lives in a path containing 'gemma'."""


_FakeGemmaModel.__module__ = "mlx_lm.models.gemma"


class _FakeMistralModel:
    """Lives in a path containing 'mistral'."""


_FakeMistralModel.__module__ = "mlx_lm.models.mistral"


class _FakeUnknownModel:
    """Module path doesn't match any known family."""


_FakeUnknownModel.__module__ = "mlx_lm.models.custom_arch_xyz"


# ── Positive detection ────────────────────────────────────────────────────────


def test_infer_llama():
    model = _FakeLlamaModel()
    assert _infer_model_family(model) == "llama"


def test_infer_gemma():
    model = _FakeGemmaModel()
    assert _infer_model_family(model) == "gemma"


def test_infer_mistral():
    """Mistral is not in SUPPORTED_FAMILIES — must return None, not 'mistral'."""
    model = _FakeMistralModel()
    assert _infer_model_family(model) is None, (
        "_infer_model_family must return None for unsupported families like 'mistral'. "
        "The upstream support gate (upgrade_cache_list) now fails-closed on None, "
        "so returning an unsupported family name would cause UnsupportedModelError."
    )


# ── Unknown family returns None ───────────────────────────────────────────────


def test_infer_unknown_returns_none():
    model = _FakeUnknownModel()
    assert _infer_model_family(model) is None


# ── Class name also triggers detection ────────────────────────────────────────


class _LlamaNamedModel:
    pass


_LlamaNamedModel.__module__ = "turboquant.custom"
_LlamaNamedModel.__name__ = "MyLlamaWrapper"


def test_infer_from_class_name():
    model = _LlamaNamedModel()
    assert _infer_model_family(model) == "llama"


# ── Return type is always str or None ────────────────────────────────────────


def test_return_type_is_str_or_none():
    for cls in [_FakeLlamaModel, _FakeUnknownModel]:
        result = _infer_model_family(cls())
        assert result is None or isinstance(result, str)
