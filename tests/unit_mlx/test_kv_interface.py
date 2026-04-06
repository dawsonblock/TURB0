# flake8: noqa

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import TurboQuantKVCache


def fake_quantize_main(x, *, config):
    return x, mx.ones(
        (*x.shape[:-1], x.shape[-1] // config.k_group_size), dtype=mx.float32
    )


def fake_dequantize_main(packed, scales, *, config):
    return packed


def test_cache_stores_generic_blocks_qjl():
    cfg = TurboQuantConfig(
        k_bits=3,
        k_group_size=32,
        v_group_size=32,
        residual_mode="qjl",
        qjl_proj_dim=64,
        qjl_seed=7,
        rotation_pad_to_pow2=True,
    )

    cache = TurboQuantKVCache(
        config=cfg,
        quantize_main=fake_quantize_main,
        dequantize_main=fake_dequantize_main,
    )

    k = mx.random.normal(shape=(2, 5, 96), key=mx.random.key(0))
    block = cache.append_keys(k)

    assert cache.num_blocks == 1
    assert block.residual.mode == "qjl"
    assert "bits" in block.residual.data
    assert "norms" in block.residual.data


def test_cache_uses_polar_quantizer_and_restores_from_state():
    cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
    cache = TurboQuantKVCache(config=cfg)

    k = mx.random.normal(shape=(1, 2, 4, 128), key=mx.random.key(2))
    block = cache.append_keys(k)
    assert block.polar is not None
    assert cache.byte_size() > 0

    state = cache.state()
    restored = TurboQuantKVCache.from_state(
        state,
        quantize_main=None,
        dequantize_main=None,
    )
    k_hat = cache.decode_block_full(0)
    restored_hat = restored.decode_block_full(0)

    assert restored.block(0).polar is not None
    assert restored_hat.shape == k_hat.shape
    diff = float(mx.max(mx.abs(restored_hat - k_hat)).item())
    assert diff < 1e-5
