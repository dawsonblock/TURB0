"""
TurboQuant state serialisation schema.

STATE_SCHEMA_VERSION is bumped whenever the dict layout produced by
``TurboQuantKVCache.state()`` changes in a backward-incompatible way.

Consumers (save/load, test fixtures, mlx-lm cache migration) must pass
the state dict through ``validate_state`` before restoring a cache.
"""

from __future__ import annotations

from typing import Any

from turboquant.config import TurboQuantConfig
from turboquant.errors import TurboQuantStateError

STATE_SCHEMA_VERSION: int = 3
"""Integer version of the TurboQuant state dict format.

Changelog
---------
1  initial  – flat tensor payload with top-level packed K/V arrays
2  legacy   – adds 11 stored config keys plus optional calibrated scales,
              while keeping the flat tensor payload
3  current  – canonical block-list payload. Stores the same top-level scalar
              and config keys, but serialises encoded key blocks under
              ``blocks``. ``validate_state()`` still accepts legacy v1/v2
              flat payloads for migration and fixture compatibility.
"""

_SUPPORTED_VERSIONS = frozenset({1, 2, 3})
_REQUIRED_SCALAR_KEYS = frozenset(
    {"schema_version", "offset", "d_head", "d_pad", "v_dim", "v_pad"}
)
_CONFIG_KEYS_V23 = frozenset(
    {
        "k_bits",
        "k_group_size",
        "v_bits",
        "v_group_size",
        "v_enabled",
        "rotation",
        "rotation_seed",
        "residual_topk",
        "scale_dtype",
        "v_scale_dtype",
        "eps",
    }
)
_BLOCK_KEYS_V3 = frozenset(
    {
        "packed_main",
        "scales",
        "residual_mode",
        "residual_data_keys",
        "d_head",
        "d_rot",
        "d_quant",
        "algorithm",
        "orig_dim",
    }
)


def _shape_token_len(arr) -> int | None:
    if arr is None or not hasattr(arr, "shape"):
        return None
    if len(arr.shape) < 3:
        return None
    return arr.shape[2]  # type: ignore


def _expect_config_match(state: dict[str, Any], config: TurboQuantConfig) -> None:
    mismatches = []
    checks = {
        "k_bits": config.k_bits,
        "k_group_size": config.k_group_size,
        "v_bits": config.v_bits,
        "v_group_size": config.v_group_size,
        "v_enabled": config.v_enabled,
        "rotation": config.rotation,
        "rotation_seed": config.rotation_seed,
        "residual_topk": config.residual_topk,
        "scale_dtype": config.scale_dtype,
        "v_scale_dtype": config.v_scale_dtype,
    }
    for key, expected in checks.items():
        if key in state and state[key] != expected:
            mismatches.append(f"{key}: state={state[key]!r}, config={expected!r}")

    if "eps" in state and abs(float(state["eps"]) - float(config.eps)) > 1e-12:
        mismatches.append(f"eps: state={state['eps']!r}, config={config.eps!r}")

    if mismatches:
        raise TurboQuantStateError(
            "State/config mismatch detected. Refusing restore because future "
            "encode behavior would diverge: " + "; ".join(mismatches)
        )


def _validate_block_payload(blocks: Any, *, offset: int) -> None:
    if not isinstance(blocks, list):
        raise TurboQuantStateError(
            "State schema v3 requires 'blocks' to be a list of serialized "
            "EncodedKeyBlock payloads."
        )

    if offset > 0 and not blocks:
        raise TurboQuantStateError(
            f"State has offset={offset} but 'blocks' is empty. State is corrupt."
        )

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise TurboQuantStateError(
                f"Block {index} must be a dict, got {type(block).__name__!r}."
            )

        missing = _BLOCK_KEYS_V3 - block.keys()
        if missing:
            raise TurboQuantStateError(
                f"Block {index} is missing required keys: {sorted(missing)}."
            )

        for key in ("packed_main", "scales"):
            value = block.get(key)
            if value is not None and not isinstance(value, str):
                raise TurboQuantStateError(
                    f"Block {index} field {key!r} must be a base64 string or None, "
                    f"got {type(value).__name__!r}."
                )

        if not isinstance(block.get("residual_mode"), str):
            raise TurboQuantStateError(
                f"Block {index} field 'residual_mode' must be a string."
            )

        residual_keys = block.get("residual_data_keys")
        if not isinstance(residual_keys, list) or not all(
            isinstance(item, str) for item in residual_keys
        ):
            raise TurboQuantStateError(
                f"Block {index} field 'residual_data_keys' must be a list of strings."
            )

        if not isinstance(block.get("algorithm"), str):
            raise TurboQuantStateError(
                f"Block {index} field 'algorithm' must be a string."
            )

        for key in ("d_head", "d_rot", "d_quant", "orig_dim"):
            value = block.get(key)
            if not isinstance(value, int) or value < 0:
                raise TurboQuantStateError(
                    f"Block {index} field {key!r} must be a non-negative int, got {value!r}."
                )


def validate_state(
    state: dict[str, Any],
    config: TurboQuantConfig | None = None,
) -> None:
    """Raise ``TurboQuantStateError`` if *state* is not a valid KV state dict."""
    if "schema_version" not in state:
        raise TurboQuantStateError(
            "State dict is missing 'schema_version'. "
            "This state was produced by an older TurboQuant version. "
            "Re-run prefill to rebuild the cache."
        )

    version = state["schema_version"]
    if not isinstance(version, int):
        raise TurboQuantStateError(
            f"'schema_version' must be an int, got {type(version).__name__!r}."
        )
    if version not in _SUPPORTED_VERSIONS:
        raise TurboQuantStateError(
            f"State schema version {version} is incompatible with the "
            f"current loader (supports {sorted(_SUPPORTED_VERSIONS)}). "
            "Re-run prefill to rebuild the cache."
        )

    missing = _REQUIRED_SCALAR_KEYS - state.keys()
    if missing:
        raise TurboQuantStateError(
            f"State dict is missing required keys: {sorted(missing)}."
        )

    offset = state["offset"]
    if not isinstance(offset, int) or offset < 0:
        raise TurboQuantStateError(
            f"'offset' must be a non-negative int, got {offset!r}."
        )

    if version >= 3:
        _validate_block_payload(state.get("blocks"), offset=offset)
    elif offset > 0:
        k_packed = state.get("k_packed")
        if k_packed is None:
            raise TurboQuantStateError(
                f"State has offset={offset} but 'k_packed' is None. State is corrupt."
            )
        token_len = _shape_token_len(k_packed)
        if token_len is not None and token_len < offset:
            raise TurboQuantStateError(
                f"'k_packed' token dimension ({token_len}) is smaller than "
                f"offset ({offset}). State is corrupt."
            )

    if version >= 2:
        missing_cfg = _CONFIG_KEYS_V23 - state.keys()
        if missing_cfg:
            raise TurboQuantStateError(
                f"State schema v{version} is missing config keys: {sorted(missing_cfg)}."
            )

    if config is None:
        return

    if version >= 2:
        _expect_config_match(state, config)

    if offset == 0:
        return

    if version < 3:
        k_scales = state.get("k_scales")
        d_pad = state.get("d_pad")
        if k_scales is not None and hasattr(k_scales, "shape") and d_pad is not None:
            ng_stored = k_scales.shape[-1]
            ng_expected = d_pad // config.k_group_size
            if ng_stored != ng_expected:
                raise TurboQuantStateError(
                    f"k_scales group count {ng_stored} does not match "
                    f"config.k_group_size={config.k_group_size} with d_pad={d_pad} "
                    f"(expected {ng_expected} groups)."
                )

        v_scales = state.get("v_scales")
        v_pad = state.get("v_pad")
        if (
            config.v_enabled
            and v_scales is not None
            and hasattr(v_scales, "shape")
            and v_pad is not None
        ):
            vg_stored = v_scales.shape[-1]
            vg_expected = v_pad // config.v_group_size
            if vg_stored != vg_expected:
                raise TurboQuantStateError(
                    f"v_scales group count {vg_stored} does not match "
                    f"config.v_group_size={config.v_group_size} with v_pad={v_pad} "
                    f"(expected {vg_expected} groups)."
                )

        if version >= 2:
            k_cal = state.get("k_calibrated_scales")
            if k_cal is not None and hasattr(k_cal, "shape") and d_pad is not None:
                expected = d_pad // config.k_group_size
                if k_cal.shape[-1] != expected:
                    raise TurboQuantStateError(
                        f"k_calibrated_scales width {k_cal.shape[-1]} does not match "
                        f"expected group count {expected}."
                    )
            v_cal = state.get("v_calibrated_scales")
            if v_cal is not None and hasattr(v_cal, "shape") and v_pad is not None:
                expected = v_pad // config.v_group_size
                if v_cal.shape[-1] != expected:
                    raise TurboQuantStateError(
                        f"v_calibrated_scales width {v_cal.shape[-1]} does not match "
                        f"expected group count {expected}."
                    )
