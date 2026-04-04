import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.core.residual_codec import build_residual_codec


def test_codec_dispatch_none():
    cfg = TurboQuantConfig(algorithm="turboquant_mse", residual_mode="none")
    codec = build_residual_codec(cfg)
    assert codec.mode == "none"


def test_codec_dispatch_topk():
    cfg = TurboQuantConfig(
        algorithm="legacy_topk",
        residual_mode="topk",
        residual_topk=8,
    )
    codec = build_residual_codec(cfg)
    assert codec.mode == "topk"


def test_codec_dispatch_qjl():
    cfg = TurboQuantConfig(residual_mode="qjl", qjl_proj_dim=32, qjl_seed=7)
    codec = build_residual_codec(cfg)
    assert codec.mode == "qjl"


def test_qjl_roundtrip_shape():
    cfg = TurboQuantConfig(residual_mode="qjl", qjl_proj_dim=32, qjl_seed=7)
    codec = build_residual_codec(cfg)

    x = mx.random.normal(shape=(2, 4, 96), key=mx.random.key(0))
    payload = codec.encode(x, config=cfg)
    xhat = codec.decode(payload, config=cfg)

    assert xhat.shape == x.shape
