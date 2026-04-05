import pytest

from turboquant.integrations.mlx.upgrade import upgrade_cache_list

# Skip on non-Apple-Silicon hosts and define mx for use in tests.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")


class MockCache:
    def __init__(self, offset=0, keys=None, values=None):
        self.offset = offset
        self.keys = keys
        self.values = values

    def byte_size(self):
        return 1024


def test_upgrade_cache_list_idempotence():
    """Upgrading an already converted cache list should be idempotent."""
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache

    batch, heads, seq, d_head = 1, 2, 8, 32
    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=3,
        k_group_size=32,
        residual_mode="none",
    )

    # Two plain KV caches with real mx.array keys/values.
    k = mx.random.normal([batch, heads, seq, d_head])
    v = mx.random.normal([batch, heads, seq, d_head])
    cache_list = [MockCache(offset=seq, keys=k, values=v) for _ in range(2)]

    events1 = upgrade_cache_list(
        cache_list,
        k_start=0,
        config=cfg,
        model_family="llama",
    )
    assert all(ev.upgraded for ev in events1), (
        "First upgrade_cache_list call must promote all eligible layers"
    )
    assert all(isinstance(c, TurboQuantKCache) for c in cache_list), (
        "All cache entries must be TurboQuantKCache after first upgrade"
    )

    events2 = upgrade_cache_list(
        cache_list,
        k_start=0,
        config=cfg,
        model_family="llama",
    )
    assert all(not ev.upgraded for ev in events2), (
        "Second upgrade_cache_list call must be idempotent: "
        "no layer should be re-upgraded"
    )

    assert all(isinstance(c, TurboQuantKCache) for c in cache_list), (
        "Cache entries must remain TurboQuantKCache "
        "after second (no-op) upgrade"
    )


def test_upgrade_cache_list_unsupported_model():
    """Unsupported model families should fail fast."""
    from turboquant.config import TurboQuantConfig
    from turboquant.errors import UnsupportedModelError

    cache_list = [MockCache(offset=100)]
    config = TurboQuantConfig()

    with pytest.raises(UnsupportedModelError):
        upgrade_cache_list(
            cache_list,
            k_start=50,
            config=config,
            model_family="unsupported_model_xyz",
        )


def test_upgrade_cache_list_supports_experimental_polar_mode():
    """PolarQuant configs should flow through the upgrade path."""
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache

    batch, heads, seq, d_head = 1, 2, 8, 128
    cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")

    k = mx.random.normal([batch, heads, seq, d_head])
    v = mx.random.normal([batch, heads, seq, d_head])
    cache_list = [MockCache(offset=seq, keys=k, values=v)]

    events = upgrade_cache_list(
        cache_list,
        k_start=0,
        config=cfg,
        model_family="llama",
    )

    assert len(events) == 1
    assert events[0].upgraded
    upgraded = cache_list[0]
    assert isinstance(upgraded, TurboQuantKCache)
    assert upgraded._impl.block(0).polar is not None
    assert upgraded._impl.state()["algorithm"] == "polarquant_exp"
