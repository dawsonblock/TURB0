"""Compatibility shim for the historical public cache-adapter import path.

Public runtime callers should use
``turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)``.
Internal callers should import from
``turboquant.integrations.mlx._cache_adapter``.
This module remains only so legacy eval helpers and tests do not break.
"""

from turboquant.integrations.mlx._cache_adapter import (
    SUPPORTED_MLX_MIN,
    TurboQuantKCache,
    bfloat16,
    concat,
    dummy_dequantize_main,
    dummy_quantize_main,
    eval_and_sync,
    float16,
    float32,
    int32,
    is_mlx_available,
    item,
    mlx_version,
    ones,
    softmax,
    to_bfloat16,
    to_float16,
    to_float32,
    uint8,
    zeros,
)

__all__ = [
    "SUPPORTED_MLX_MIN",
    "TurboQuantKCache",
    "bfloat16",
    "concat",
    "dummy_dequantize_main",
    "dummy_quantize_main",
    "eval_and_sync",
    "float16",
    "float32",
    "int32",
    "is_mlx_available",
    "item",
    "mlx_version",
    "ones",
    "softmax",
    "to_bfloat16",
    "to_float16",
    "to_float32",
    "uint8",
    "zeros",
]
