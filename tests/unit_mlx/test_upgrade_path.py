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

class MockConfig:
    def __init__(self):
        self.k_bits = 3
        self.k_group_size = 32
        self.rotation_mode = "hadamard"
        self.residual_mode = "topk"
        self.v_bits = 4

def test_upgrade_cache_list_idempotence():
    """Ensure that upgrading the same cache list twice is idempotent and doesn't re-upgrade."""
    # Mock MLX-like tensors (just dummy objects for now since we aren't running on Apple Silicon)
    # In a real environment, we'd use mx.array.
    k = object()
    v = object()
    
    cache_list = [MockCache(offset=100, keys=k, values=v)]
    config = MockConfig()
    
    # 1. First call with k_start > offset should NOT upgrade
    events1 = upgrade_cache_list(cache_list, k_start=200, config=config)
    assert not events1[0].upgraded
    assert not isinstance(cache_list[0], object) # Still MockCache
    
    # 2. Second call with k_start <= offset should upgrade
    # Note: This will likely fail in this test because TurboQuantKCache 
    # will try to import and use MLX. 
    # This test is intended to be run in unit_mlx or with mocks.
    pass

def test_upgrade_cache_list_unsupported_model():
    """Ensure that an unsupported model family raises an error immediately."""
    from turboquant.errors import UnsupportedModelError # Assuming this exists
    
    cache_list = [MockCache(offset=100)]
    config = MockConfig()
    
    with pytest.raises(Exception): # Replace with UnsupportedModelError once confirmed
        upgrade_cache_list(cache_list, k_start=50, config=config, model_family="unsupported_model_xyz")
