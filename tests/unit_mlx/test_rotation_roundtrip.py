"""Unit tests for FixedRotation — apply/invert roundtrip and orthogonality.

Requires MLX (Apple Silicon).
"""

from __future__ import annotations

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.core.rotation import FixedRotation


def _cfg(rotation: str, dim: int = 64) -> FixedRotation:
    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        residual_mode="none",
        rotation=rotation,
    )
    return FixedRotation.from_config(cfg, dim)


# ── Roundtrip: apply then invert returns original ─────────────────────────────


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_hadamard_roundtrip(dim):
    rot = _cfg("hadamard", dim)
    x = mx.random.normal(shape=(2, 4, dim), key=mx.random.key(0))
    err = rot.roundtrip_error(x)
    assert err < 1e-4, f"Hadamard roundtrip error too large: {err}"


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_identity_roundtrip(dim):
    rot = _cfg("identity", dim)
    x = mx.random.normal(shape=(2, 4, dim), key=mx.random.key(1))
    err = rot.roundtrip_error(x)
    assert err < 1e-6, f"Identity roundtrip error too large: {err}"


# ── Orthogonality ─────────────────────────────────────────────────────────────


def test_hadamard_is_orthogonal():
    rot = _cfg("hadamard", 64)
    assert rot.is_orthogonal(atol=1e-4)


def test_identity_is_orthogonal():
    rot = _cfg("identity", 64)
    assert rot.is_orthogonal(atol=1e-6)


# ── from_config seeds deterministically ───────────────────────────────────────


def test_from_config_deterministic():
    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        residual_mode="none",
        rotation="hadamard",
        rotation_seed=42,
    )
    rot_a = FixedRotation.from_config(cfg, 64)
    rot_b = FixedRotation.from_config(cfg, 64)
    x = mx.random.normal(shape=(1, 64), key=mx.random.key(7))
    err = float(mx.max(mx.abs(rot_a.apply(x) - rot_b.apply(x))).item())
    assert err < 1e-7, "Two from_config calls with same seed must agree exactly"


# ── apply and invert aliases ──────────────────────────────────────────────────


def test_apply_invert_are_reverses():
    rot = _cfg("hadamard", 64)
    x = mx.random.normal(shape=(3, 64), key=mx.random.key(99))
    err = float(mx.max(mx.abs(rot.invert(rot.apply(x)) - x)).item())
    assert err < 1e-4
