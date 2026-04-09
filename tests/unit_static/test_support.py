"""
tests/unit_static/test_support.py — model-family support gate tests.

Covers ``turboquant.runtime.support``:
  * ``is_supported_model_family`` — membership check
  * ``assert_supported_model_family`` — raises UnsupportedModelError on reject
  * normalisation logic (case folding, numeric suffix stripping, underscore split)

No MLX required.
"""

from __future__ import annotations

import pytest

from turboquant.errors import UnsupportedModelError
from turboquant.runtime.support import (
    SUPPORTED_FAMILIES,
    assert_supported_model_family,
    is_supported_model_family,
)


# ── SUPPORTED_FAMILIES constant ───────────────────────────────────────────────


def test_supported_families_contains_llama() -> None:
    assert "llama" in SUPPORTED_FAMILIES


def test_supported_families_contains_gemma() -> None:
    assert "gemma" in SUPPORTED_FAMILIES


def test_supported_families_is_frozenset() -> None:
    assert isinstance(SUPPORTED_FAMILIES, frozenset)


# ── is_supported_model_family — accepted names ────────────────────────────────


def test_llama_is_supported() -> None:
    assert is_supported_model_family("llama") is True


def test_gemma_is_supported() -> None:
    assert is_supported_model_family("gemma") is True


def test_llama_upper_is_supported() -> None:
    assert is_supported_model_family("LLAMA") is True


def test_gemma_mixed_case_is_supported() -> None:
    assert is_supported_model_family("Gemma") is True


def test_llama3_numeric_suffix_is_supported() -> None:
    # Numeric suffix is stripped before allowlist lookup.
    assert is_supported_model_family("llama3") is True


def test_llama_underscore_variant_is_supported() -> None:
    # Split on '_' and take the first segment.
    assert is_supported_model_family("llama_3") is True


def test_gemma2_is_supported() -> None:
    assert is_supported_model_family("gemma2") is True


# ── is_supported_model_family — rejected names ────────────────────────────────


def test_gpt_is_not_supported() -> None:
    assert is_supported_model_family("gpt") is False


def test_mistral_is_not_supported() -> None:
    assert is_supported_model_family("mistral") is False


def test_phi_is_not_supported() -> None:
    assert is_supported_model_family("phi") is False


def test_empty_string_is_not_supported() -> None:
    assert is_supported_model_family("") is False


def test_none_normalises_to_unsupported() -> None:
    # None gets stringified to "none", which is not in the allowlist.
    assert is_supported_model_family("none") is False


# ── assert_supported_model_family — accepted names ────────────────────────────


def test_assert_llama_does_not_raise() -> None:
    assert_supported_model_family("llama")


def test_assert_gemma_does_not_raise() -> None:
    assert_supported_model_family("gemma")


def test_assert_llama3_does_not_raise() -> None:
    assert_supported_model_family("llama3")


# ── assert_supported_model_family — rejected names ────────────────────────────


def test_assert_gpt_raises_unsupported_model_error() -> None:
    with pytest.raises(UnsupportedModelError):
        assert_supported_model_family("gpt")


def test_assert_mistral_raises_unsupported_model_error() -> None:
    with pytest.raises(UnsupportedModelError):
        assert_supported_model_family("mistral")


def test_assert_unsupported_error_message_includes_name() -> None:
    with pytest.raises(UnsupportedModelError, match="phi"):
        assert_supported_model_family("phi")


def test_assert_unsupported_error_message_mentions_supported_families() -> None:
    with pytest.raises(UnsupportedModelError, match="gemma|llama"):
        assert_supported_model_family("mixtral")


def test_assert_unsupported_model_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        assert_supported_model_family("unknown_model")
