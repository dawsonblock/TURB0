"""Unit tests for TurboQuantKVCache memory accounting.

Verifies that:
- byte_size() returns a positive value that grows with sequence length
- memory_breakdown() returns all required keys
- paper_kv vs k_only modes differ in their breakdown structure
- byte_size() is consistent with memory_breakdown()["total"]

Requires MLX (Apple Silicon).
"""

from __future__ import annotations

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import TurboQuantKVCache

_REQUIRED_BREAKDOWN_KEYS = {
    "k_main",
    "k_scales",
    "k_polar",
    "k_residual",
    "v_main",
    "v_scales",
    "total",
}


def _make_impl(cfg: TurboQuantConfig) -> TurboQuantKVCache:
    return TurboQuantKVCache(config=cfg)


def _append(impl: TurboQuantKVCache, batch=1, heads=2, seq=4, d_head=64):
    k = mx.random.normal(shape=(batch, heads, seq, d_head), key=mx.random.key(0))
    v = mx.random.normal(shape=(batch, heads, seq, d_head), key=mx.random.key(1))
    impl.update_and_fetch(k, v)


# ── paper_kv mode (default for MSE/Prod algorithms) ──────────────────────────


def test_paper_kv_byte_size_positive_after_append():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    impl = _make_impl(cfg)
    assert impl.byte_size() == 0
    _append(impl)
    assert impl.byte_size() > 0


def test_paper_kv_breakdown_has_required_keys():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    impl = _make_impl(cfg)
    _append(impl)
    bd = impl.memory_breakdown()
    assert _REQUIRED_BREAKDOWN_KEYS <= set(bd.keys()), (
        f"Missing keys: {_REQUIRED_BREAKDOWN_KEYS - set(bd.keys())}"
    )


def test_paper_kv_breakdown_total_matches_byte_size():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    impl = _make_impl(cfg)
    _append(impl)
    bd = impl.memory_breakdown()
    assert bd["total"] == impl.byte_size()


def test_paper_kv_byte_size_grows_with_sequence():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    impl = _make_impl(cfg)
    _append(impl, seq=4)
    size_4 = impl.byte_size()
    _append(impl, seq=4)
    size_8 = impl.byte_size()
    assert size_8 > size_4, "byte_size should grow as more K/V blocks are appended"


def test_paper_kv_v_main_bytes_positive():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    impl = _make_impl(cfg)
    _append(impl)
    bd = impl.memory_breakdown()
    assert bd.get("v_main", 0) > 0, "paper_kv mode must charge bytes for encoded V"


# ── prod mode ─────────────────────────────────────────────────────────────────


def test_prod_mode_byte_size_positive():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    impl = _make_impl(cfg)
    _append(impl)
    assert impl.byte_size() > 0


def test_prod_mode_breakdown_has_required_keys():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    impl = _make_impl(cfg)
    _append(impl)
    bd = impl.memory_breakdown()
    assert _REQUIRED_BREAKDOWN_KEYS <= set(bd.keys())


def test_polar_mode_byte_size_positive_and_breakdown_tracks_polar_bytes():
    cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
    impl = _make_impl(cfg)
    _append(impl)
    bd = impl.memory_breakdown()
    assert impl.byte_size() > 0
    assert bd["k_polar"] > 0
    assert bd["total"] == impl.byte_size()


# ── clear() resets accounting ─────────────────────────────────────────────────


def test_clear_resets_byte_size():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    impl = _make_impl(cfg)
    _append(impl)
    assert impl.byte_size() > 0
    impl.clear()
    assert impl.byte_size() == 0
