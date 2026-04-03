import sys

import pytest


def _block_mlx():
    """Block mlx imports and return the previously cached modules for later restoration."""
    saved = {
        key: sys.modules[key]
        for key in ("mlx", "mlx.core")
        if key in sys.modules
    }
    sys.modules["mlx"] = None  # type: ignore[assignment]
    sys.modules["mlx.core"] = None  # type: ignore[assignment]
    return saved


def _restore_mlx(saved: dict):
    """Restore mlx sys.modules entries from a previously captured snapshot."""
    for key in ("mlx", "mlx.core"):
        if key in saved:
            sys.modules[key] = saved[key]
        else:
            sys.modules.pop(key, None)


def test_import_turboquant():
    """Verify turboquant imports cleanly without MLX."""
    # Temporarily hide mlx if it is installed
    saved = _block_mlx()

    try:
        import turboquant

        assert turboquant.__version__ is not None

        from turboquant.config import TurboQuantConfig

        assert TurboQuantConfig is not None

    finally:
        _restore_mlx(saved)


def test_mlx_import_error_message():
    """Verify the right error is raised when lazily loading MLX dependencies."""
    saved = _block_mlx()

    try:
        import turboquant

        with pytest.raises(ImportError):
            # Accessing an MLX-dependent symbol while mlx is blocked must raise
            # ImportError, not silently return a stub.
            _ = turboquant.KVCompressor
    finally:
        _restore_mlx(saved)
