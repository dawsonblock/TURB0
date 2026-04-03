import pytest
import os
from turboquant.integrations.mlx.upgrade import upgrade_cache_list, CacheUpgradeEvent


class MockCache:
    def __init__(self, offset=0, keys=None, values=None):
        self.offset = offset
        self.keys = keys
        self.values = values

    def byte_size(self):
        return 1024


@pytest.mark.skip(
    reason="Not implemented: second upgrade leg requires real mx.array objects "
    "(TurboQuantKCache calls _dc_replace on a real TurboQuantConfig, then "
    "update_and_fetch via MLX). Run on Apple Silicon with MLX installed."
)
def test_upgrade_cache_list_idempotence():
    """Ensure that upgrading the same cache list twice is idempotent and doesn't re-upgrade."""
    raise NotImplementedError


def test_upgrade_cache_list_unsupported_model():
    """Ensure that an unsupported model family raises UnsupportedModelError immediately."""
    from turboquant.errors import UnsupportedModelError
    from turboquant.config import TurboQuantConfig

    cache_list = [MockCache(offset=100)]
    config = TurboQuantConfig()

    with pytest.raises(UnsupportedModelError):
        upgrade_cache_list(cache_list, k_start=50, config=config, model_family="unsupported_model_xyz")
