# flake8: noqa

"""Unit tests for QJLProjector inner-product estimation.

The QJL estimator uses 1-bit sign sketches: it captures the *angle* between
vectors and scales by ||k||/||q||.  Because of the nonlinear 1-bit
quantisation, the estimate is not an unbiased estimator of the raw inner
product q·k = ||q||·||k||·cos(θ).

Instead we test the properties that DO hold:
- Output shape is correct.
- ``estimate_inner_product`` is an alias of ``dot_estimate``.
- Estimates are POSITIVELY CORRELATED with true inner products across random
  vector pairs (sign agreement ≥ 65 % — well above the 50 % chance baseline).
- The estimator responds to scale: doubling ||k|| doubles the estimate.

Requires MLX (Apple Silicon).
"""

from __future__ import annotations

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.core.qjl import QJLProjector

# ── estimate_inner_product is an alias of dot_estimate ────────────────────────


def test_estimate_inner_product_alias():
    qjl = QJLProjector(proj_dim=64, seed=7)
    q = mx.random.normal(shape=(2, 3, 96), key=mx.random.key(0))
    k = mx.random.normal(shape=(2, 3, 96), key=mx.random.key(1))

    bits, norms, meta = qjl.encode(k)
    via_old = qjl.dot_estimate(q, bits, norms, meta)
    via_new = qjl.estimate_inner_product(q, bits, norms, meta)

    diff = float(mx.max(mx.abs(via_old - via_new)).item())
    assert diff < 1e-6, "estimate_inner_product differs from dot_estimate"


# ── Output shape ──────────────────────────────────────────────────────────────


def test_estimate_inner_product_shape():
    qjl = QJLProjector(proj_dim=64, seed=42)
    q = mx.random.normal(shape=(2, 5, 96), key=mx.random.key(0))
    k = mx.random.normal(shape=(2, 7, 96), key=mx.random.key(1))

    bits, norms, meta = qjl.encode(k)
    est = qjl.estimate_inner_product(q, bits, norms, meta)
    assert est.shape == (2, 5, 7), f"Expected (2,5,7), got {est.shape}"


# ── Sign agreement: estimates and true inner products agree in sign > chance ──


def test_sign_correlation_above_chance():
    """Estimates must agree in sign with true inner products > 65 % of the time."""
    d = 128
    n_pairs = 200
    qjl = QJLProjector(proj_dim=d, seed=0)
    agreements = 0
    for i in range(n_pairs):
        q = mx.random.normal(shape=(1, 1, d), key=mx.random.key(i * 2))
        k = mx.random.normal(shape=(1, 1, d), key=mx.random.key(i * 2 + 1))
        true_dot = float(mx.sum(q * k).item())
        bits, norms, meta = qjl.encode(k)
        est = float(qjl.estimate_inner_product(q, bits, norms, meta)[0, 0, 0].item())
        if (est >= 0) == (true_dot >= 0):
            agreements += 1

    rate = agreements / n_pairs
    assert rate >= 0.65, (
        f"Sign agreement rate {rate:.2%} below 65 % threshold — "
        "QJL estimates are not correlated with true inner products"
    )


# ── Scale sensitivity: doubling ||k|| doubles the estimate ────────────────────


def test_estimate_scales_with_k_norm():
    """QJL estimate scales linearly with ||k|| (norm stored explicitly)."""
    qjl = QJLProjector(proj_dim=64, seed=5)
    q = mx.random.normal(shape=(1, 1, 64), key=mx.random.key(0))
    k = mx.random.normal(shape=(1, 1, 64), key=mx.random.key(1))
    k2 = k * 2.0  # double the norm

    bits1, norms1, meta1 = qjl.encode(k)
    bits2, norms2, meta2 = qjl.encode(k2)
    est1 = float(qjl.estimate_inner_product(q, bits1, norms1, meta1)[0, 0, 0].item())
    est2 = float(qjl.estimate_inner_product(q, bits2, norms2, meta2)[0, 0, 0].item())

    if abs(est1) > 1e-4:
        ratio = est2 / est1
        assert 1.5 <= ratio <= 2.5, (
            f"Doubling ||k|| should ~double the estimate; got ratio={ratio:.3f}"
        )
