"""
Legacy compatibility shim for turboquant.integrations.mlx.cache_adapter.
Deprecated: Use 'from turboquant.integrations.mlx.cache_adapter import ...' instead.
"""

import warnings

from turboquant.integrations.mlx.cache_adapter import *  # noqa: F403

warnings.warn(
    "Importing from 'integrations.mlx.cache_adapter' is deprecated. "
    "Use 'turboquant.integrations.mlx.cache_adapter' instead.",
    DeprecationWarning,
    stacklevel=2,
)
