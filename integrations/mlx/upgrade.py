"""
Legacy compatibility shim for turboquant.integrations.mlx.upgrade.
Deprecated: Use 'from turboquant.integrations.mlx.upgrade import ...' instead.
"""

import warnings

from turboquant.integrations.mlx.upgrade import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'integrations.mlx.upgrade' is deprecated. "
    "Use 'turboquant.integrations.mlx.upgrade' instead.",
    DeprecationWarning,
    stacklevel=2,
)
