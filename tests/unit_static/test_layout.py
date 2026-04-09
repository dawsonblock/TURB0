"""
tests/unit_static/test_layout.py — ensure_layout contract tests.

Verifies the shape-enforcement behaviour of
``turboquant.runtime.layout.ensure_layout``.

No MLX required — uses a minimal array stub that exposes ``.ndim``
and ``.shape``.
"""

from __future__ import annotations

import pytest

from turboquant.runtime.layout import ensure_layout


class _FakeArray:
    """Minimal array stub satisfying the ensure_layout interface."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


# ── Valid inputs ──────────────────────────────────────────────────────────────


def test_ensure_layout_accepts_valid_4d_tensor() -> None:
    x = _FakeArray((1, 8, 16, 64))
    result = ensure_layout(x)
    assert result is x


def test_ensure_layout_returns_same_object() -> None:
    x = _FakeArray((2, 4, 32, 128))
    assert ensure_layout(x) is x


def test_ensure_layout_accepts_single_batch_single_head() -> None:
    x = _FakeArray((1, 1, 1, 1))
    ensure_layout(x)


def test_ensure_layout_accepts_large_tensor() -> None:
    x = _FakeArray((4, 32, 2048, 128))
    ensure_layout(x)


# ── Wrong number of dimensions ────────────────────────────────────────────────


def test_ensure_layout_rejects_3d_tensor() -> None:
    x = _FakeArray((8, 16, 64))
    with pytest.raises(ValueError, match="4-D"):
        ensure_layout(x)


def test_ensure_layout_rejects_2d_tensor() -> None:
    x = _FakeArray((16, 64))
    with pytest.raises(ValueError, match="4-D"):
        ensure_layout(x)


def test_ensure_layout_rejects_1d_tensor() -> None:
    x = _FakeArray((64,))
    with pytest.raises(ValueError, match="4-D"):
        ensure_layout(x)


def test_ensure_layout_rejects_5d_tensor() -> None:
    x = _FakeArray((1, 8, 16, 64, 2))
    with pytest.raises(ValueError, match="4-D"):
        ensure_layout(x)


# ── Zero dimensions ───────────────────────────────────────────────────────────


def test_ensure_layout_rejects_zero_batch() -> None:
    x = _FakeArray((0, 8, 16, 64))
    with pytest.raises(ValueError, match="All dimensions"):
        ensure_layout(x)


def test_ensure_layout_rejects_zero_heads() -> None:
    x = _FakeArray((1, 0, 16, 64))
    with pytest.raises(ValueError, match="All dimensions"):
        ensure_layout(x)


def test_ensure_layout_rejects_zero_seq_len() -> None:
    x = _FakeArray((1, 8, 0, 64))
    with pytest.raises(ValueError, match="All dimensions"):
        ensure_layout(x)


def test_ensure_layout_rejects_zero_head_dim() -> None:
    x = _FakeArray((1, 8, 16, 0))
    with pytest.raises(ValueError, match="All dimensions"):
        ensure_layout(x)


# ── Custom name in error message ──────────────────────────────────────────────


def test_ensure_layout_includes_name_in_error_message() -> None:
    x = _FakeArray((16, 64))
    with pytest.raises(ValueError, match="my_keys"):
        ensure_layout(x, name="my_keys")


def test_ensure_layout_default_name_in_error_message() -> None:
    x = _FakeArray((16, 64))
    with pytest.raises(ValueError, match="tensor"):
        ensure_layout(x)
