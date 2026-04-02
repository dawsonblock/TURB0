"""
TurboQuant — research-stage KV-cache compression for selected MLX/Apple-Silicon RLM paths.

This package exposes the supported public surface for the current TurboQuant prototype.
Do not treat the package-level API as production-certified unless the corresponding
runtime-certification artifacts have been generated on Apple Silicon.

Public API
+--------
TurboQuantConfig          — runtime-immutable configuration
TurboQuantPipeline        — low-level encode/decode pipeline
TurboQuantKVCache         — canonical KV cache implementation
TurboQuantKCache          — MLX cache-adapter interface
KVCompressor              — compatibility alias to TurboQuantKVCache
calibrate                 — calibration pass over representative data
upgrade_cache_list        — utility to migrate MLX cache lists
"""

from turboquant._deps import (
    check_mlx_version,
    has_mlx,
    is_apple_silicon,
    require_mlx,
)
from turboquant.config import TurboQuantConfig

# Validate MLX version bounds at import time (no-op if MLX is absent)
check_mlx_version()

# Lazy imports for MLX-dependent runtime symbols
_MLX_DEPENDENT = frozenset(
    {
        "calibrate",
        "TurboQuantPipeline",
        "TurboQuantKVCache",
        "TurboQuantKCache",
        "upgrade_cache_list",
        "KVCompressor",
    }
)


class _MLXNotAvailableStub:
    """Sentinel returned by __getattr__ when MLX is absent.

    Returning a stub (rather than raising) lets ``hasattr()`` succeed on
    non-Apple machines so that static API-surface tests can verify that the
    symbols exist in the public namespace.  Attempting to *call* the stub
    still raises a descriptive RuntimeError.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<MLXNotAvailableStub: {self._name}>"

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"'{self._name}' requires MLX on Apple Silicon. "
            "TurboQuant inference is only supported on macOS + Apple Silicon."
        )

    # Attribute access on the stub itself should also be descriptive.
    def __getattr__(self, item: str):
        raise RuntimeError(
            f"Cannot access '{self._name}.{item}': MLX is not available."
        )


def __getattr__(name: str):
    if name not in _MLX_DEPENDENT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if not has_mlx():
        if is_apple_silicon():
            # MLX is explicitly absent/blocked on an Apple Silicon machine.
            # Raise ImportError so callers (and test_import_smoke) get a clear message.
            raise ImportError(
                f"TurboQuant: '{name}' requires the `mlx` package. "
                "Install it with: pip install 'turboquant[apple]'"
            )
        # Non-Apple platform: MLX will never be available.  Return a stub so
        # that hasattr() and __all__ inspection work in static test environments.
        return _MLXNotAvailableStub(name)

    if name == "calibrate":
        from turboquant.calibration.fit_quantizer import calibrate
        return calibrate
    elif name == "TurboQuantPipeline":
        from turboquant.core.pipeline import TurboQuantPipeline
        return TurboQuantPipeline
    elif name == "TurboQuantKVCache":
        from turboquant.runtime.kv_interface import TurboQuantKVCache
        return TurboQuantKVCache
    elif name == "TurboQuantKCache":
        from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
        return TurboQuantKCache
    elif name == "upgrade_cache_list":
        from turboquant.integrations.mlx.upgrade import upgrade_cache_list
        return upgrade_cache_list
    elif name == "KVCompressor":
        # compatibility alias for TurboQuantKVCache
        from turboquant.runtime.kv_interface import TurboQuantKVCache as KVCompressor
        return KVCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TurboQuantConfig",
    "TurboQuantPipeline",
    "TurboQuantKVCache",
    "TurboQuantKCache",
    "KVCompressor",
    "calibrate",
    "upgrade_cache_list",
    "check_mlx_version",
    "has_mlx",
    "is_apple_silicon",
    "require_mlx",
]

__version__ = "0.2.2"
