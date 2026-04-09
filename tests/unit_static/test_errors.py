"""
tests/unit_static/test_errors.py — error class hierarchy and behaviour.

Verifies that every custom exception in turboquant.errors:
  * can be instantiated with a message
  * inherits from the documented base classes
  * can be caught by both its own type and its base(s)

No MLX required.
"""

from __future__ import annotations

import pytest

from turboquant.errors import (
    CompressionFailureError,
    TurboQuantCompatibilityError,
    TurboQuantConfigError,
    TurboQuantError,
    TurboQuantKernelError,
    TurboQuantShapeError,
    TurboQuantStateError,
    UnsupportedModelError,
)


# ── Hierarchy ─────────────────────────────────────────────────────────────────


def test_turbo_quant_error_is_exception() -> None:
    assert issubclass(TurboQuantError, Exception)


def test_config_error_inherits_turbo_quant_error() -> None:
    assert issubclass(TurboQuantConfigError, TurboQuantError)


def test_config_error_inherits_value_error() -> None:
    assert issubclass(TurboQuantConfigError, ValueError)


def test_shape_error_inherits_turbo_quant_error() -> None:
    assert issubclass(TurboQuantShapeError, TurboQuantError)


def test_shape_error_inherits_value_error() -> None:
    assert issubclass(TurboQuantShapeError, ValueError)


def test_state_error_inherits_turbo_quant_error() -> None:
    assert issubclass(TurboQuantStateError, TurboQuantError)


def test_state_error_inherits_value_error() -> None:
    assert issubclass(TurboQuantStateError, ValueError)


def test_kernel_error_inherits_turbo_quant_error() -> None:
    assert issubclass(TurboQuantKernelError, TurboQuantError)


def test_kernel_error_inherits_runtime_error() -> None:
    assert issubclass(TurboQuantKernelError, RuntimeError)


def test_compatibility_error_inherits_turbo_quant_error() -> None:
    assert issubclass(TurboQuantCompatibilityError, TurboQuantError)


def test_compatibility_error_inherits_type_error() -> None:
    assert issubclass(TurboQuantCompatibilityError, TypeError)


def test_unsupported_model_error_inherits_turbo_quant_error() -> None:
    assert issubclass(UnsupportedModelError, TurboQuantError)


def test_unsupported_model_error_inherits_value_error() -> None:
    assert issubclass(UnsupportedModelError, ValueError)


def test_compression_failure_error_inherits_turbo_quant_error() -> None:
    assert issubclass(CompressionFailureError, TurboQuantError)


def test_compression_failure_error_inherits_runtime_error() -> None:
    assert issubclass(CompressionFailureError, RuntimeError)


# ── Instantiation and message passing ─────────────────────────────────────────


def test_turbo_quant_error_message() -> None:
    err = TurboQuantError("base error")
    assert "base error" in str(err)


def test_config_error_message() -> None:
    err = TurboQuantConfigError("bad config value")
    assert "bad config value" in str(err)


def test_shape_error_message() -> None:
    err = TurboQuantShapeError("dimension mismatch")
    assert "dimension mismatch" in str(err)


def test_state_error_message() -> None:
    err = TurboQuantStateError("corrupt state")
    assert "corrupt state" in str(err)


def test_kernel_error_message() -> None:
    err = TurboQuantKernelError("kernel unsupported")
    assert "kernel unsupported" in str(err)


def test_compatibility_error_message() -> None:
    err = TurboQuantCompatibilityError("adapter drift")
    assert "adapter drift" in str(err)


def test_unsupported_model_error_message() -> None:
    err = UnsupportedModelError("model not in allowlist")
    assert "model not in allowlist" in str(err)


def test_compression_failure_error_message() -> None:
    err = CompressionFailureError("NaN in K scales")
    assert "NaN in K scales" in str(err)


# ── Catch-by-base-class ───────────────────────────────────────────────────────


def test_config_error_caught_as_turbo_quant_error() -> None:
    with pytest.raises(TurboQuantError):
        raise TurboQuantConfigError("config")


def test_config_error_caught_as_value_error() -> None:
    with pytest.raises(ValueError):
        raise TurboQuantConfigError("config")


def test_state_error_caught_as_value_error() -> None:
    with pytest.raises(ValueError):
        raise TurboQuantStateError("state")


def test_kernel_error_caught_as_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        raise TurboQuantKernelError("kernel")


def test_unsupported_model_error_caught_as_turbo_quant_error() -> None:
    with pytest.raises(TurboQuantError):
        raise UnsupportedModelError("model")


def test_compression_failure_error_caught_as_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        raise CompressionFailureError("failure")


def test_compatibility_error_caught_as_type_error() -> None:
    with pytest.raises(TypeError):
        raise TurboQuantCompatibilityError("compat")
