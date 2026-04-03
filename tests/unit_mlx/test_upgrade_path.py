import pytest

# Platform gate: skip on non-Apple-Silicon hosts and define mx for use in tests.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

import os
from turboquant.integrations.mlx.upgrade import upgrade_cache_list, CacheUpgradeEvent


class MockCache:
    def __init__(self, offset=0, keys=None, values=None):
        self.offset = offset
        self.keys = keys
        self.values = values

    def byte_size(self):
        return 1024


def test_upgrade_cache_list_idempotence():
    """Upgrading a cache list that is already TurboQuantKCache must not re-upgrade any layer."""
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list

    batch, heads, seq, d_head = 1, 2, 8, 32
    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )

    # Two plain KV caches with real mx.array keys/values and offset above k_start=0.
    k = mx.random.normal([batch, heads, seq, d_head])
    v = mx.random.normal([batch, heads, seq, d_head])
    cache_list = [MockCache(offset=seq, keys=k, values=v) for _ in range(2)]

    # First call: both layers are above threshold → both must be upgraded.
    events1 = upgrade_cache_list(cache_list, k_start=0, config=cfg, model_family=None)
    assert all(ev.upgraded for ev in events1), (
        "First upgrade_cache_list call must promote all eligible layers"
    )
    assert all(isinstance(c, TurboQuantKCache) for c in cache_list), (
        "All cache entries must be TurboQuantKCache after first upgrade"
    )

    # Second call: every layer is already TurboQuantKCache → none must be re-upgraded.
    events2 = upgrade_cache_list(cache_list, k_start=0, config=cfg, model_family=None)
    assert all(not ev.upgraded for ev in events2), (
        "Second upgrade_cache_list call must be idempotent: no layer should be re-upgraded"
    )

    # Cache list must still contain only TurboQuantKCache instances.
    assert all(isinstance(c, TurboQuantKCache) for c in cache_list), (
        "Cache entries must remain TurboQuantKCache after second (no-op) upgrade"
    )


def test_upgrade_cache_list_unsupported_model():
    """Ensure that an unsupported model family raises UnsupportedModelError immediately."""
    from turboquant.errors import UnsupportedModelError
    from turboquant.config import TurboQuantConfig

    cache_list = [MockCache(offset=100)]
    config = TurboQuantConfig()

    with pytest.raises(UnsupportedModelError):
        upgrade_cache_list(cache_list, k_start=50, config=config, model_family="unsupported_model_xyz")
