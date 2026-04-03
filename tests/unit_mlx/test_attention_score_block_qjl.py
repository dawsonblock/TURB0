import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import encode_k_block
from turboquant.runtime.attention import score_block


def fake_quantize_main(x, *, config):
    return x, mx.ones((*x.shape[:-1], x.shape[-1] // config.k_group_size), dtype=mx.float32)


def fake_dequantize_main(packed, scales, *, config):
    return packed


def test_score_block_qjl_shape():
    cfg = TurboQuantConfig(
        k_bits=3,
        k_group_size=32, v_group_size=32,
        residual_mode="qjl",
        qjl_proj_dim=64,
        qjl_seed=7,
        rotation_pad_to_pow2=True,
    )

    k = mx.random.normal(shape=(2, 5, 96), key=mx.random.key(0))
    block = encode_k_block(
        k,
        config=cfg,
        quantize_main=fake_quantize_main,
        dequantize_main=fake_dequantize_main,
    )

    q_rot = mx.random.normal(shape=(2, 3, block.d_rot), key=mx.random.key(1))
    scores = score_block(
        q_rot,
        block,
        config=cfg,
        dequantize_main=fake_dequantize_main,
    )

    assert scores.shape == (2, 3, 5)
