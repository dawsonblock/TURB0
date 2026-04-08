"""Central model-family support gate for runtime imports."""

from __future__ import annotations

from turboquant.errors import UnsupportedModelError

SUPPORTED_FAMILIES: frozenset[str] = frozenset({"llama", "gemma"})


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
            f"Model family {name!r} is not supported by TurboQuant. "
            f"Supported families: {sorted(SUPPORTED_FAMILIES)}."
        )
