"""turboquant.runtime — lazy facade; does not import MLX at module level."""

from __future__ import annotations

__all__ = ["TurboQuantKVCache", "ensure_layout"]


def __getattr__(name: str):
    if name == "TurboQuantKVCache":
        from turboquant.runtime.kv_interface import TurboQuantKVCache
        return TurboQuantKVCache
    if name == "ensure_layout":
        from turboquant.runtime.layout import ensure_layout
        return ensure_layout
    raise AttributeError(f"module 'turboquant.runtime' has no attribute {name!r}")
