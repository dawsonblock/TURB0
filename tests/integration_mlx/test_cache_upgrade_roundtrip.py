"""
Structural test: upgrade_cache_list must promote a cache entry to
TurboQuantKCache when its offset exceeds k_start, and the promoted
cache must correctly service subsequent update_and_fetch calls.

Uses synthetic MLX tensors; no model weights are required.
Auto-skips on non-MLX machines via pytest.importorskip.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

from turboquant.config import TurboQuantConfig
from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
from turboquant.integrations.mlx.upgrade import upgrade_cache_list
from turboquant.runtime.kv_interface import TurboQuantKeysView


class _MinimalKVCache:
    """Minimal stand-in for mlx_lm KVCache with the attributes upgrade_cache_list reads."""

    def __init__(self, keys: "mx.array", values: "mx.array"):
        self.keys = keys
        self.values = values
        self.offset = keys.shape[-2]


def test_cache_upgrade_roundtrip():
    """
    After upgrade_cache_list promotes a layer, subsequent update_and_fetch
    must return TurboQuantKeysView (not a dense array).
    """
    batch, heads, seq, d_head = 1, 2, 8, 32
    k_start = 4  # upgrade threshold — offset=8 is above it

    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )

    # Start with two "dense" cache layers (offset already above threshold)
    k0 = mx.random.normal([batch, heads, seq, d_head])
    v0 = mx.random.normal([batch, heads, seq, d_head])
    k1 = mx.random.normal([batch, heads, seq, d_head])
    v1 = mx.random.normal([batch, heads, seq, d_head])
    prompt_cache = [_MinimalKVCache(k0, v0), _MinimalKVCache(k1, v1)]

    events = upgrade_cache_list(
        prompt_cache, k_start=k_start, config=cfg, model_family="llama"
    )

    # Both layers must have been upgraded
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    for ev in events:
        assert ev.upgraded, (
            f"Layer {ev.layer_index} was not upgraded (offset={ev.offset_at_upgrade}, "
            f"k_start={k_start})"
        )
        assert ev.new_type == "TurboQuantKCache", (
            f"Layer {ev.layer_index} new_type={ev.new_type!r}, expected 'TurboQuantKCache'"
        )

    # Both entries in prompt_cache must now be TurboQuantKCache instances
    for i, cache in enumerate(prompt_cache):
        assert isinstance(cache, TurboQuantKCache), (
            f"prompt_cache[{i}] is {type(cache).__name__}, expected TurboQuantKCache"
        )

    # A follow-on decode step must return TurboQuantKeysView
    k_new = mx.random.normal([batch, heads, 1, d_head])
    v_new = mx.random.normal([batch, heads, 1, d_head])
    k_out, v_out = prompt_cache[0].update_and_fetch(k_new, v_new)

    assert isinstance(k_out, TurboQuantKeysView), (
        f"Post-upgrade update_and_fetch returned {type(k_out).__name__}, "
        "expected TurboQuantKeysView."
    )
    assert isinstance(v_out, mx.array), (
        "Post-upgrade update_and_fetch must return values as mx.array."
    )


def test_cache_upgrade_below_threshold_skipped():
    """Layers whose offset is below k_start must NOT be upgraded."""
    batch, heads, seq, d_head = 1, 2, 2, 32
    k_start = 8  # seq=2 < k_start=8 → no upgrade

    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )

    k0 = mx.random.normal([batch, heads, seq, d_head])
    v0 = mx.random.normal([batch, heads, seq, d_head])
    prompt_cache = [_MinimalKVCache(k0, v0)]

    events = upgrade_cache_list(
        prompt_cache, k_start=k_start, config=cfg, model_family="llama"
    )

    assert len(events) == 1
    assert not events[0].upgraded, (
        f"Layer was upgraded unexpectedly (offset={events[0].offset_at_upgrade}, "
        f"k_start={k_start})"
    )
    assert not isinstance(prompt_cache[0], TurboQuantKCache), (
        "Cache below threshold must not be promoted to TurboQuantKCache."
    )


def test_cache_upgrade_idempotent():
    """Calling upgrade_cache_list twice must not double-upgrade a layer."""
    batch, heads, seq, d_head = 1, 2, 8, 32

    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )

    k0 = mx.random.normal([batch, heads, seq, d_head])
    v0 = mx.random.normal([batch, heads, seq, d_head])
    prompt_cache = [_MinimalKVCache(k0, v0)]

    # First call — should upgrade
    events1 = upgrade_cache_list(
        prompt_cache, k_start=4, config=cfg, model_family="llama"
    )
    assert events1[0].upgraded

    # Second call — already TurboQuantKCache, must be a no-op
    events2 = upgrade_cache_list(
        prompt_cache, k_start=4, config=cfg, model_family="llama"
    )
    assert not events2[0].upgraded, (
        "Second upgrade_cache_list call must not re-upgrade an already-upgraded layer."
    )
