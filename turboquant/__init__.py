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
def __getattr__(name: str):
    if name == "calibrate":
        require_mlx("calibrate()")
        from turboquant.calibration.fit_quantizer import calibrate
        return calibrate
    elif name == "TurboQuantPipeline":
        require_mlx("TurboQuantPipeline")
        from turboquant.core.pipeline import TurboQuantPipeline
        return TurboQuantPipeline
    elif name == "TurboQuantKVCache":
        require_mlx("TurboQuantKVCache")
        from turboquant.runtime.kv_interface import TurboQuantKVCache
        return TurboQuantKVCache
    elif name == "TurboQuantKCache":
        require_mlx("TurboQuantKCache")
        from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
        return TurboQuantKCache
    elif name == "upgrade_cache_list":
        require_mlx("upgrade_cache_list()")
        from turboquant.integrations.mlx.upgrade import upgrade_cache_list
        return upgrade_cache_list
    elif name == "KVCompressor":
        # compatibility alias for TurboQuantKVCache
        require_mlx("KVCompressor")
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
