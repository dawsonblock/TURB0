"""
Structural proof: TurboQuantKCache.update_and_fetch() must return
TurboQuantKeysView as the key output — never a plain mx.array.
A raw mx.array would mean the dense (uncompressed) fallback was taken.

This test uses synthetic MLX tensors; no model weights are required.
It will auto-skip on machines without MLX (non-Apple Silicon CI).
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

from turboquant.config import TurboQuantConfig
from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
from turboquant.runtime.kv_interface import TurboQuantKeysView


def test_path_not_dense_fallback():
    """
    Verify update_and_fetch returns TurboQuantKeysView, not a dense mx.array.

    This is the primary structural invariant of the TurboQuant KV-cache path.
    If this test fails, attention dispatch will route to the uncompressed path
    and no memory savings will be realised.
    """
    cfg = TurboQuantConfig(
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )
    cache = TurboQuantKCache(cfg)

    # Shape: [batch=1, heads=2, seq=4, d_head=32]
    # d_head=32 is a clean multiple of k_group_size=32 — no padding required.
    keys = mx.zeros([1, 2, 4, 32])
    values = mx.zeros([1, 2, 4, 32])

    k_out, v_out = cache.update_and_fetch(keys, values)

    assert isinstance(k_out, TurboQuantKeysView), (
        f"Expected TurboQuantKeysView from update_and_fetch, "
        f"got {type(k_out).__name__}. "
        "TurboQuant is falling back to the dense (uncompressed) key path."
    )
    # Belt-and-suspenders: a TurboQuantKeysView must not itself be a raw array.
    assert not isinstance(k_out, mx.array), (
        "Key output must not be a plain mx.array (dense fallback detected)."
    )
    # Values are returned dense by design (V-compression is optional).
    assert isinstance(v_out, mx.array), (
        "Values must be returned as a plain mx.array."
    )
    # Offset bookkeeping: 4 tokens were appended starting at position 0.
    assert k_out.start == 0, f"Expected start=0, got {k_out.start}"
    assert k_out.end == 4, f"Expected end=4, got {k_out.end}"
