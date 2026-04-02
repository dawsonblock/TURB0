import os
import pytest
import turboquant

def test_public_api_surface():
    """Ensure that the turboquant package exports the correct canonical names."""
    
    # Canonical exports
    assert hasattr(turboquant, "TurboQuantConfig")
    assert hasattr(turboquant, "TurboQuantPipeline")
    assert hasattr(turboquant, "TurboQuantKVCache")
    assert hasattr(turboquant, "TurboQuantKCache")
    assert hasattr(turboquant, "upgrade_cache_list")
    assert hasattr(turboquant, "calibrate")
    
    # Legacy alias
    assert hasattr(turboquant, "KVCompressor")
    
    # Check __all__
    # Note: Using set comparison for equality
    expected_all = {
        "TurboQuantConfig",
        "TurboQuantPipeline",
        "TurboQuantKVCache",
        "TurboQuantKCache",
        "upgrade_cache_list",
        "KVCompressor",
        "calibrate",
        "check_mlx_version",
        "has_mlx",
        "is_apple_silicon",
        "require_mlx",
    }
    
    # Depending on how __all__ is handled with __getattr__
    current_all = set(turboquant.__all__)
    for name in expected_all:
        assert name in current_all, f"{name} should be in __all__"

def test_removed_turboquant_runtime():
    """Verify that TurboQuantRuntime is removed from the supported surface and raises an error."""
    # It shouldn't be in __all__
    assert "TurboQuantRuntime" not in turboquant.__all__
    
    from turboquant.runtime.api import TurboQuantRuntime
    
    with pytest.raises(RuntimeError) as excinfo:
        TurboQuantRuntime()
        
    assert "prototype and has been removed" in str(excinfo.value)
    assert "TurboQuantKCache" in str(excinfo.value)
