# flake8: noqa

"""Deterministic bias snapshot for the paper-facing two-stage path.

This lane does not claim that the repo has already proven an unbiased
inner-product estimator. It makes the current paper-facing comparison explicit
and checkable instead:

- `paper_mse` gives the scalar-only baseline.
- `paper_prod_qjl` gives the scalar main stage plus the QJL residual score
  contribution used by runtime attention.

The test records stable synthetic-workload statistics for signed error,
absolute error, variance, and a tail metric. The assertions are intentionally
about measurability and bounded behavior, not about proving paper equivalence.
"""

from __future__ import annotations

import math

import pytest

from tests.helpers.mlx_env import MLX_SKIP_MARKER

pytestmark = MLX_SKIP_MARKER

mx = pytest.importorskip("mlx.core")

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.core.rotation import FixedRotation
from turboquant.runtime.attention import score_block


def _synthetic_queries_and_keys():
    q = mx.random.normal(shape=(8, 12, 128), key=mx.random.key(0))
    k = mx.random.normal(shape=(8, 16, 128), key=mx.random.key(1))
    return q, k


def _true_scores(config: TurboQuantConfig, q, k):
    rotation = FixedRotation.from_config(config, int(q.shape[-1]))
    q_rot = rotation.apply(q.astype(mx.float32))
    k_rot = rotation.apply(k.astype(mx.float32))
    return q_rot @ mx.swapaxes(k_rot, -1, -2)


def _score_stats(config: TurboQuantConfig):
    q, k = _synthetic_queries_and_keys()
    pipe = TurboQuantPipeline(config)
    block = pipe.encode_k(k)
    estimated = score_block(
        q,
        block,
        config=config,
        dequantize_main=pipe._get_k_quant().dequantize,
    )
    true = _true_scores(config, q, k)

    error = estimated - true
    flat_error = error.reshape(-1)
    flat_abs_error = mx.abs(error).reshape(-1)
    mean_abs_true = float(mx.mean(mx.abs(true)).item())
    q95_index = max(int(flat_abs_error.shape[0] * 0.95) - 1, 0)
    q95_abs_error = float(mx.sort(flat_abs_error)[q95_index].item())

    return {
        "mean_signed_error": float(mx.mean(flat_error).item()),
        "mean_abs_error": float(mx.mean(flat_abs_error).item()),
        "error_variance": float(mx.var(flat_error).item()),
        "q95_abs_error": q95_abs_error,
        "normalized_mean_bias": float(mx.mean(flat_error).item()) / mean_abs_true,
        "normalized_mean_abs_error": float(mx.mean(flat_abs_error).item())
        / mean_abs_true,
    }


def test_paper_prod_score_block_matches_two_stage_sum():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    q, k = _synthetic_queries_and_keys()
    pipe = TurboQuantPipeline(cfg)
    block = pipe.encode_k(k)

    combined = score_block(
        q,
        block,
        config=cfg,
        dequantize_main=pipe._get_k_quant().dequantize,
    )

    rotation = FixedRotation.from_config(cfg, int(q.shape[-1]))
    q_rot = rotation.apply(q.astype(mx.float32))
    main_hat = pipe._get_k_quant().dequantize(
        block.packed_main,
        block.scales,
        config=cfg,
    )
    main_scores = q_rot @ mx.swapaxes(main_hat[..., : block.d_rot], -1, -2)

    from turboquant.core.residual_codec import build_residual_codec

    residual_scores = build_residual_codec(cfg).dot_estimate(
        q_rot,
        block.residual,
        config=cfg,
    )
    manual = main_scores + residual_scores

    diff = float(mx.max(mx.abs(combined - manual)).item())
    assert diff < 1e-5, "paper_prod_qjl score_block no longer matches the two-stage sum"


def test_inner_product_bias_lane_is_bounded_and_distinct():
    mse_stats = _score_stats(TurboQuantConfig.from_preset("paper_mse"))
    prod_stats = _score_stats(TurboQuantConfig.from_preset("paper_prod"))

    for stats in (mse_stats, prod_stats):
        for value in stats.values():
            assert math.isfinite(value), f"non-finite bias metric: {stats}"
        assert abs(stats["normalized_mean_bias"]) < 0.05
        assert 0.0 < stats["normalized_mean_abs_error"] < 0.5
        assert stats["error_variance"] > 0.0
        assert stats["q95_abs_error"] >= stats["mean_abs_error"]

    delta = abs(
        prod_stats["normalized_mean_abs_error"]
        - mse_stats["normalized_mean_abs_error"]
    )
    assert delta > 0.01, (
        "paper_mse and paper_prod_qjl should produce distinguishable score-error "
        "profiles on the fixed synthetic workload"
    )