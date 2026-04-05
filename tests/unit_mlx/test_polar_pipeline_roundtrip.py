"""Unit tests for PolarQuant pipeline encode/decode and block persistence.

Requires MLX (Apple Silicon).
"""

from __future__ import annotations

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import EncodedKeyBlock, TurboQuantPipeline
from turboquant.core.polar_quant import PolarQuantizer


def _polar_config() -> TurboQuantConfig:
    return TurboQuantConfig.polarquant_exp(
        k_bits=3,
        k_group_size=64,
        v_bits=4,
        v_group_size=64,
        rotation="random_orthogonal",
    )


def test_polar_pipeline_uses_polar_quantizer() -> None:
    pipe = TurboQuantPipeline(_polar_config())
    assert isinstance(pipe._k_quant, PolarQuantizer)


def test_polar_encode_decode_shape_and_finite_error() -> None:
    pipe = TurboQuantPipeline(_polar_config())
    x = mx.random.normal(shape=(2, 4, 96), key=mx.random.key(7))

    block = pipe.encode_k(x)
    x_hat = pipe.decode_k(block)

    assert block.polar is not None
    assert x_hat.shape == x.shape
    rel_err = float((mx.mean(mx.abs(x_hat - x)) / (mx.mean(mx.abs(x)) + 1e-6)).item())
    assert rel_err < 1.25


def test_polar_encoded_block_serialisation_roundtrip() -> None:
    pipe = TurboQuantPipeline(_polar_config())
    x = mx.random.normal(shape=(1, 2, 128), key=mx.random.key(11))

    block = pipe.encode_k(x)
    payload = block.to_dict()
    restored = EncodedKeyBlock.from_dict(payload)
    x_hat = pipe.decode_k(restored)

    assert restored.polar is not None
    assert x_hat.shape == x.shape
    assert restored.byte_size() > 0
