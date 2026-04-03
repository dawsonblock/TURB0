"""
Structural test: turboquant_streaming_attention must produce
a finite, correctly-shaped output.

The quantized path is lossy by design, so this test does NOT assert
numerical equivalence to dense SDPA within a tight tolerance.
Instead it verifies:
  1. Output shape matches the query shape.
  2. No NaN or Inf values are produced.
  3. The function accepts the TurboQuantKeysView protocol without raising.

This uses synthetic MLX tensors; no model weights are required.
Auto-skips on non-MLX machines via pytest.importorskip.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

from turboquant.config import TurboQuantConfig
from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
from turboquant.runtime.attention import turboquant_streaming_attention
from turboquant.runtime.kv_interface import TurboQuantKeysView


def _build_cache_with_keys(cfg, batch, heads, seq, d_head):
    """Helper: populate a TurboQuantKCache and return (cache, keys_view)."""
    cache = TurboQuantKCache(cfg)
    keys = mx.random.normal([batch, heads, seq, d_head])
    values = mx.random.normal([batch, heads, seq, d_head])
    keys_view, _ = cache.update_and_fetch(keys, values)
    return cache, keys_view


def test_streaming_attention_equivalence():
    """
    turboquant_streaming_attention must return a finite tensor with the
    correct shape: [batch, q_heads, q_seq, d_head].
    """
    batch, kv_heads, kv_seq, d_head = 1, 2, 8, 32
    q_heads = kv_heads  # no GQA in this test

    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )
    cache, keys_view = _build_cache_with_keys(cfg, batch, kv_heads, kv_seq, d_head)

    queries = mx.random.normal([batch, q_heads, 1, d_head])
    scale = 1.0 / (d_head ** 0.5)

    out = turboquant_streaming_attention(queries, keys_view, scale=scale)
    mx.eval(out)

    assert out.shape == (batch, q_heads, 1, d_head), (
        f"Expected shape {(batch, q_heads, 1, d_head)}, got {tuple(out.shape)}"
    )
    assert not bool(mx.any(mx.isnan(out)).item()), (
        "turboquant_streaming_attention produced NaN values."
    )
    assert not bool(mx.any(mx.isinf(out)).item()), (
        "turboquant_streaming_attention produced Inf values."
    )


def test_streaming_attention_multi_block():
    """
    With two separate update_and_fetch calls (two encoded blocks),
    the output must still be finite and correctly shaped.
    """
    batch, kv_heads, d_head = 1, 2, 32

    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )
    cache = TurboQuantKCache(cfg)

    # Two separate key blocks
    k1 = mx.random.normal([batch, kv_heads, 4, d_head])
    v1 = mx.random.normal([batch, kv_heads, 4, d_head])
    cache.update_and_fetch(k1, v1)

    k2 = mx.random.normal([batch, kv_heads, 4, d_head])
    v2 = mx.random.normal([batch, kv_heads, 4, d_head])
    keys_view, _ = cache.update_and_fetch(k2, v2)

    queries = mx.random.normal([batch, kv_heads, 1, d_head])
    scale = 1.0 / (d_head ** 0.5)

    out = turboquant_streaming_attention(queries, keys_view, scale=scale)
    mx.eval(out)

    assert out.shape == (batch, kv_heads, 1, d_head), (
        f"Multi-block: expected shape {(batch, kv_heads, 1, d_head)}, got {tuple(out.shape)}"
    )
    assert not bool(mx.any(mx.isnan(out)).item()), "Multi-block produced NaN."
    assert not bool(mx.any(mx.isinf(out)).item()), "Multi-block produced Inf."
