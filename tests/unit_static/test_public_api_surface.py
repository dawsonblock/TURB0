import pytest

import turboquant

# MLX-dependent symbols: accessing these raises ImportError when MLX is absent.
# Test their presence via __all__ (a static list) rather than hasattr(), which
# would silently return False when ImportError is raised.
_MLX_DEPENDENT = {
    "TurboQuantPipeline",
    "TurboQuantKVCache",
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


def test_internal_adapter_not_exported_at_package_root():
    """TurboQuantKCache should no longer be advertised from the package root."""
    assert "TurboQuantKCache" not in turboquant.__all__
    assert not hasattr(turboquant, "TurboQuantKCache")


def test_package_root_docstring_matches_public_contract():
    """The package-root docstring must reflect the same narrow contract as the docs."""
    doc = turboquant.__doc__ or ""

    assert "RLM" not in doc, "turboquant.__doc__ still contains the 'RLM' typo."
    assert "LLM" in doc, (
        "turboquant.__doc__ must describe the supported MLX/Apple-Silicon LLM paths."
    )
    assert "passing full-model" in doc and "runtime-certification artifacts" in doc, (
        "turboquant.__doc__ must not imply that any partial artifact set makes the package production-certified."
    )
    assert "internal/eval-only" in doc, (
        "turboquant.__doc__ must label TurboQuantKCache as an internal/eval-only adapter."
    )
