# flake8: noqa

"""Unit tests for TurboQuantPipeline MSE encode/decode roundtrip.

Verifies that:
- Shape is preserved after encode → decode
- The LloydMaxScalarQuantizer is selected in MSE/Prod mode
- Reconstruction error is within a reasonable bound (not identity —
  3-bit quantisation introduces noise by design)

Requires MLX (Apple Silicon).
"""

from __future__ import annotations

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.core.quantizer import LloydMaxScalarQuantizer


def _mse_config(k_bits=3, k_group_size=64):
    return TurboQuantConfig(
        algorithm="turboquant_mse",
        residual_mode="none",
        k_bits=k_bits,
        k_group_size=k_group_size,
        rotation="hadamard",
    )


# ── Pipeline selects LloydMax for MSE mode ────────────────────────────────────


def test_mse_pipeline_uses_lloydmax():
    cfg = _mse_config()
    pipe = TurboQuantPipeline(cfg)
    assert isinstance(pipe._k_quant, LloydMaxScalarQuantizer)


# ── Encode → decode preserves shape ──────────────────────────────────────────


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 64),  # [batch, heads, d_head]
        (2, 4, 128),
        (1, 8, 32),
    ],
)
def test_encode_decode_shape(shape):
    cfg = _mse_config(k_bits=3, k_group_size=shape[-1])
    pipe = TurboQuantPipeline(cfg)
    x = mx.random.normal(shape=shape, key=mx.random.key(0))

    block = pipe.encode_k(x)
    x_hat = pipe.decode_k(block)
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} != {x.shape}"


# ── Reconstruction is close (not an identity, bounded by quant error) ─────────


@pytest.mark.parametrize(
    "k_bits,max_rel_err",
    [
        (4, 0.20),  # 4-bit Lloyd-Max: ~10-15 % relative error expected
        (3, 0.35),  # 3-bit: moderate error
        (2, 0.60),  # 2-bit: coarser
        (1, 1.30),  # 1-bit: sign-only, high error by design
    ],
)
def test_encode_decode_bounded_error(k_bits, max_rel_err):
    cfg = _mse_config(k_bits=k_bits, k_group_size=64)
    pipe = TurboQuantPipeline(cfg)
    x = mx.random.normal(shape=(4, 2, 64), key=mx.random.key(k_bits))

    block = pipe.encode_k(x)
    x_hat = pipe.decode_k(block)

    rel_err = float((mx.mean(mx.abs(x_hat - x)) / (mx.mean(mx.abs(x)) + 1e-6)).item())
    assert rel_err < max_rel_err, (
        f"k_bits={k_bits}: relative error {rel_err:.4f} exceeds bound {max_rel_err}"
    )


# ── block.algorithm matches config ───────────────────────────────────────────────


def test_encoded_block_stores_algorithm():
    cfg = _mse_config()
    pipe = TurboQuantPipeline(cfg)
    x = mx.random.normal(shape=(1, 2, 64), key=mx.random.key(0))
    block = pipe.encode_k(x)
    assert block.algorithm == cfg.algorithm_family()


# ── to_dict / from_dict roundtrip ────────────────────────────────────────────


def test_encoded_block_serialisation():
    cfg = _mse_config()
    pipe = TurboQuantPipeline(cfg)
    x = mx.random.normal(shape=(1, 2, 64), key=mx.random.key(0))
    block = pipe.encode_k(x)

    from turboquant.core.pipeline import EncodedKeyBlock

    d = block.to_dict()
    block2 = EncodedKeyBlock.from_dict(d)
    assert block2.orig_dim == block.orig_dim
    assert block2.algorithm == block.algorithm
