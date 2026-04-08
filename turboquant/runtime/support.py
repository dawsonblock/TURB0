"""
turboquant.runtime.support — central model-family support gate.

This module is the single source of truth for which model families have
TurboQuant attention wiring and runtime-certification coverage.

The allowlist itself is generated from ``turboquant/contract.json`` at build
time so imports do not perform file I/O on the runtime hot path.
"""

from __future__ import annotations

from turboquant.errors import UnsupportedModelError
from turboquant.runtime._generated_support import SUPPORTED_FAMILIES


def _normalize(name: str) -> str:
    """Normalise a raw model family string for allowlist lookup."""
    return name.lower().split("_")[0].rstrip("0123456789")


def is_supported_model_family(name: str) -> bool:
    """Return ``True`` if *name* belongs to a TurboQuant-supported family."""
    return _normalize(name) in SUPPORTED_FAMILIES


def assert_supported_model_family(name: str) -> None:
    """Raise when *name* is not in the supported allowlist."""
    if not is_supported_model_family(name):
        raise UnsupportedModelError(
            f"Model family {name!r} is not supported by TurboQuant.  "
            f"Supported families: {sorted(SUPPORTED_FAMILIES)}.  "
            "To add support, wire the attention layer, add runtime-cert "
            "coverage, regenerate turboquant.runtime._generated_support, "
            "then update the support docs."
        )
