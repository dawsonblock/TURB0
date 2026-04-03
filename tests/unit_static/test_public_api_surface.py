import os
import pytest
import turboquant


# MLX-dependent symbols: accessing these raises ImportError when MLX is absent.
# Test their presence via __all__ (a static list) rather than hasattr(), which
# would silently return False when ImportError is raised.
_MLX_DEPENDENT = {
    "TurboQuantPipeline",
    "TurboQuantKVCache",
    "TurboQuantKCache",
    "upgrade_cache_list",
    "calibrate",
    "KVCompressor",
}


def test_public_api_surface():
    """Ensure that the turboquant package exports the correct canonical names."""

    # Non-MLX symbols — always accessible.
    assert hasattr(turboquant, "TurboQuantConfig")

    # MLX-dependent symbols — verify via __all__ (does not trigger __getattr__).
    for name in _MLX_DEPENDENT:
        assert name in turboquant.__all__, f"{name} must be listed in __all__"

    # Check __all__ contains the full expected set.
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
