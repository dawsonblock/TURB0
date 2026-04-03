import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.core.qjl import QJLProjector


def test_qjl_encode_shapes():
    qjl = QJLProjector(proj_dim=64, seed=123)
    x = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(0))

    bits, norms, meta = qjl.encode(x)

    assert bits.shape == (3, 5, 64)
    assert norms.shape == (3, 5, 1)
    assert meta.input_dim == 96
    assert meta.proj_dim == 64
    assert meta.seed == 123


def test_qjl_decode_shape():
    qjl = QJLProjector(proj_dim=64, seed=123)
    x = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(0))

    bits, norms, meta = qjl.encode(x)
    xhat = qjl.decode(bits, norms, meta)

    assert xhat.shape == x.shape


def test_qjl_dot_estimate_shape():
    qjl = QJLProjector(proj_dim=64, seed=123)
    q = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(1))
    r = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(2))

    bits, norms, meta = qjl.encode(r)
    est = qjl.dot_estimate(q, bits, norms, meta)

    assert est.shape == (3, 5, 5)

