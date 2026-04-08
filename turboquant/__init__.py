"""
TurboQuant — research-stage KV-cache compression for selected
MLX/Apple-Silicon LLM paths.

This package exposes the supported public surface for the current
TurboQuant prototype. Do not treat the package-level API as
production-certified unless passing full-model runtime-certification artifacts have been generated on Apple Silicon.

Public API
----------
TurboQuantConfig          — runtime-immutable configuration
TurboQuantPipeline        — low-level encode/decode pipeline
TurboQuantKVCache         — canonical KV cache implementation
KVCompressor              — compatibility alias for TurboQuantKVCache;
                            prefer TurboQuantKVCache for new code
calibrate                 — calibration pass over representative data
upgrade_cache_list        — canonical support-gated runtime path for
                            allowlisted MLX cache upgrades

Internal adapter note
---------------------
`TurboQuantKCache(...)` is an internal/eval-only mlx_lm adapter. It
intentionally bypasses the support gate and is therefore not exported
from the package root.
"""

from typing import TYPE_CHECKING, Any

from turboquant._deps import (
    check_mlx_version,
    has_mlx,
    is_apple_silicon,
    require_mlx,
)
from turboquant.config import TurboQuantConfig
from turboquant.patch import apply_mlx_lm_patches

if TYPE_CHECKING:
    from turboquant.calibration.fit_quantizer import calibrate
    from turboquant.core.pipeline import TurboQuantPipeline
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list
    from turboquant.runtime.kv_interface import TurboQuantKVCache
    from turboquant.runtime.kv_interface import TurboQuantKVCache as KVCompressor

# Validate MLX version bounds at import time (no-op if MLX is absent)
check_mlx_version()
apply_mlx_lm_patches()

# Lazy imports for MLX-dependent runtime symbols
_MLX_DEPENDENT = frozenset(
    {
        "calibrate",
        "TurboQuantPipeline",
        "TurboQuantKVCache",
        "upgrade_cache_list",
        "KVCompressor",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _MLX_DEPENDENT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if not has_mlx():
        # MLX is unavailable on this machine (either non-Apple platform or
        # MLX not installed on Apple Silicon).  Raise a clear ImportError
        # regardless of platform so the contract is uniform: accessing any
        # MLX-dependent symbol without MLX always fails loudly.
        raise ImportError(
            f"TurboQuant: '{name}' requires the `mlx` package on Apple "
            "Silicon. "
            "Install it with: pip install 'turboquant[apple]'"
        )

    if name == "calibrate":
        from turboquant.calibration.fit_quantizer import calibrate

        return calibrate
    elif name == "TurboQuantPipeline":
        from turboquant.core.pipeline import TurboQuantPipeline

        return TurboQuantPipeline
    elif name == "TurboQuantKVCache":
        from turboquant.runtime.kv_interface import TurboQuantKVCache

        return TurboQuantKVCache
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
    "KVCompressor",  # compatibility alias — prefer TurboQuantKVCache for new code
    "calibrate",
    "upgrade_cache_list",
    "check_mlx_version",
    "has_mlx",
    "is_apple_silicon",
    "require_mlx",
]

__version__ = "0.2.2"
