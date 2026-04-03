import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import TurboQuantKVCache


def fake_quantize_main(x, *, config):
    return x, mx.ones((*x.shape[:-1], x.shape[-1] // config.k_group_size), dtype=mx.float32)


def fake_dequantize_main(packed, scales, *, config):
    return packed


def test_cache_stores_generic_blocks_qjl():
    cfg = TurboQuantConfig(
        k_bits=3,
        k_group_size=32, v_group_size=32,
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
