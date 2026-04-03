"""
turboquant.integrations.mlx.adapter — MLX integration layer.

This module provides the MLX adapter and helper utilities for TurboQuant.
Note: MLX imports appear in multiple TurboQuant modules (kv_interface,
layout, pipeline, etc.) and are deferred to avoid breaking static analysis
on non-Apple platforms.  This file is the canonical home for adapter-level
MLX helpers (version checking, dtype casting, eval/sync).

Why an adapter module?
-----------------------
* **Testability** — unit tests that mock this module can exercise the full
  TurboQuant stack without a physical Apple Silicon GPU.
* **Compatibility** — version checks and dtype helpers live here; callers
  never inspect ``mx.__version__`` directly.

What lives here
---------------
``mlx_version()``          — ``mx.__version__`` as a string
``is_mlx_available()``     — safe existence check (no import error on non-Apple)
``eval_and_sync(*arrays)`` — ``mx.eval`` + ``mx.synchronize``; prefer over bare ``mx.eval``
``zeros(shape, dtype)``    — ``mx.zeros``
``ones(shape, dtype)``     — ``mx.ones``
``to_float32(arr)``        — cast to float32
``to_float16(arr)``        — cast to float16
``to_bfloat16(arr)``       — cast to bfloat16
``softmax(arr, axis)``     — ``nn.softmax`` via mlx.nn
``concat(arrays, axis)``   — ``mx.concatenate``
``item(arr)``              — scalar extraction (wraps ``arr.item()``)
``SUPPORTED_MLX_MIN``      — minimum supported MLX version string
"""

from __future__ import annotations

import importlib.util
import logging
import mlx.core as mx

from mlx_lm.models.cache import _BaseCache

from turboquant.config import TurboQuantConfig


class TurboQuantKCache(_BaseCache):
    """
    Internal/eval-only MLX-LM adapter for TurboQuant compression.

    Production callers should prefer ``upgrade_cache_list(...)``, which
    enforces the model-family support gate before constructing this adapter.
    Direct construction is kept for eval and compatibility helpers.
    """
    def __init__(self, config: TurboQuantConfig):
        # We don't call super().__init__() because we manage our own storage
        self.config = config
        from turboquant.runtime.kv_interface import TurboQuantKVCache

        # Let TurboQuantKVCache auto-select the quantizer based on algorithm
        # (LloydMax for paper modes, GroupScalar for experimental/legacy).
        self._impl = TurboQuantKVCache(config=config)
        self.offset = 0
        # v_cache kept for k_only legacy mode; empty in paper mode.
        self.v_cache: list = []

    @property
    def nbytes(self):
        # impl.byte_size() covers K + V (paper mode uses v_blocks; k_only uses v_cache).
        return self._impl.byte_size()

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # Implements the MLX-LM cache-adapter protocol.
        from turboquant.runtime.kv_interface import TurboQuantKeysView

        if self._impl.storage_mode == "paper_kv":
            # Delegate entirely to impl (handles V encoding into v_blocks).
            start = self._impl._offset
            self._impl.update_and_fetch(keys, values)
            self.offset = self._impl._offset
            return TurboQuantKeysView(self, start, self.offset), values
        else:
            # k_only legacy path: compress K, store dense V on adapter.
            self._impl.append_keys(keys)
            start = self.offset
            self.offset += keys.shape[2]
            self.v_cache.append(values)
            return TurboQuantKeysView(self, start, self.offset), values

    @property
    def state(self):
        return self._impl.state()

    @property
    def meta_state(self):
        return (self.config.__dict__, self.offset)


logger = logging.getLogger("turboquant.integrations.mlx.adapter")

# Minimum MLX version that TurboQuant officially supports.
# See docs/support_matrix.md and turboquant/_deps.py.
SUPPORTED_MLX_MIN: str = "0.30.0"


def is_mlx_available() -> bool:
    """Return ``True`` if ``mlx`` can be imported on this machine."""
    return importlib.util.find_spec("mlx") is not None


def mlx_version() -> str:
    """Return the installed MLX version string (e.g. ``'0.16.1'``).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If the installed version is below ``SUPPORTED_MLX_MIN``.
    """
    import mlx.core as mx  # noqa: PLC0415 (lazy import is intentional)

    ver = str(mx.__version__)
    _check_version(ver)
    return ver


def _check_version(ver: str) -> None:
    """Raise ``RuntimeError`` if *ver* is below ``SUPPORTED_MLX_MIN``."""
    try:
        from packaging.version import Version  # type: ignore[import-untyped]

        if Version(ver) < Version(SUPPORTED_MLX_MIN):
            raise RuntimeError(
                f"TurboQuant requires MLX >= {SUPPORTED_MLX_MIN}, "
                f"found {ver}.  Upgrade with: pip install --upgrade mlx"
            )
    except ImportError:
        # packaging not installed — skip version check, emit warning
        logger.warning(
            "MLX version check skipped: 'packaging' not installed.  "
            "Install it with: pip install packaging"
        )


# ── Array construction ────────────────────────────────────────────────────────


def zeros(shape: tuple[int, ...], dtype=None):
    """Create a zero-filled array of *shape* and optional *dtype*."""
    import mlx.core as mx

    kw = {"dtype": dtype} if dtype is not None else {}
    return mx.zeros(shape, **kw)


def ones(shape: tuple[int, ...], dtype=None):
    """Create a one-filled array of *shape* and optional *dtype*."""
    import mlx.core as mx

    kw = {"dtype": dtype} if dtype is not None else {}
    return mx.ones(shape, **kw)


# ── Type casting ──────────────────────────────────────────────────────────────


def to_float32(arr):
    """Cast *arr* to ``mx.float32``."""
    import mlx.core as mx

    return arr.astype(mx.float32)


def to_float16(arr):
    """Cast *arr* to ``mx.float16``."""
    import mlx.core as mx

    return arr.astype(mx.float16)


def to_bfloat16(arr):
    """Cast *arr* to ``mx.bfloat16``."""
    import mlx.core as mx

    return arr.astype(mx.bfloat16)


# ── Evaluation and synchronisation ───────────────────────────────────────────


def eval_and_sync(*arrays) -> None:
    """Evaluate all pending graphs for *arrays* and synchronise the stream.

    Prefer this over bare ``mx.eval`` in TurboQuant production code so that
    timing measurements and memory footprint readings are accurate.
    """
    import mlx.core as mx

    mx.eval(*arrays)
    mx.synchronize()


# ── Math helpers ──────────────────────────────────────────────────────────────


def softmax(arr, axis: int = -1):
    """Apply softmax along *axis* using ``mlx.core.softmax``."""
    import mlx.core as mx

    return mx.softmax(arr, axis=axis)


def concat(arrays: list, axis: int = 0):
    """Concatenate a list of arrays along *axis*."""
    import mlx.core as mx

    return mx.concatenate(arrays, axis=axis)


def item(arr) -> float | int:
    """Extract a scalar Python value from a single-element MLX array."""
    return arr.item()


# ── Dtype helpers ─────────────────────────────────────────────────────────────


def float32():
    """Return ``mx.float32``."""
    import mlx.core as mx

    return mx.float32


def float16():
    """Return ``mx.float16``."""
    import mlx.core as mx

    return mx.float16


def bfloat16():
    """Return ``mx.bfloat16``."""
    import mlx.core as mx

    return mx.bfloat16


def uint8():
    """Return ``mx.uint8``."""
    import mlx.core as mx

    return mx.uint8


def int32():
    """Return ``mx.int32``."""
    import mlx.core as mx

    return mx.int32


# ── Benchmark helpers ─────────────────────────────────────────────────────────


def dummy_quantize_main(x, *, config):
    """Trivial quantizer for benchmarks: cast to uint8 (no real compression)."""
    import mlx.core as mx

    group_size = config.k_group_size
    *prefix, d = x.shape
    d_pad = ((d + group_size - 1) // group_size) * group_size
    if d_pad > d:
        zeros = mx.zeros((*prefix, d_pad - d), dtype=x.dtype)
        x = mx.concatenate([x, zeros], axis=-1)
    n_groups = d_pad // group_size
    x_groups = x.reshape(*prefix, n_groups, group_size)
    scales = mx.abs(x_groups).max(axis=-1, keepdims=True)  # [..., n_groups, 1]
    scales = mx.maximum(scales, mx.array(1e-6, dtype=scales.dtype))
    # Pack as float16 — "dummy", not real bit-packing
    packed = x_groups.astype(mx.float16)
    return packed, scales.squeeze(-1)


def dummy_dequantize_main(packed, scales, *, config):
    """Trivial dequantizer matching dummy_quantize_main."""
    import mlx.core as mx

    *prefix, n_groups, group_size = packed.shape
    x_groups = packed.astype(mx.float32) * scales[..., None].astype(mx.float32)
    return x_groups.reshape(*prefix, n_groups * group_size)
