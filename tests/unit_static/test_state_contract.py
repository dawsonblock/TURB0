import pytest

from turboquant.config import TurboQuantConfig
from turboquant.errors import TurboQuantStateError
from turboquant.runtime.state import STATE_SCHEMA_VERSION, validate_state


def _config_fields(config: TurboQuantConfig) -> dict[str, object]:
    return {
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
        "eps": config.eps,
    }


def _canonical_v3_state() -> tuple[dict[str, object], TurboQuantConfig]:
    config = TurboQuantConfig()
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "offset": 4,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        **_config_fields(config),
        "blocks": [
            {
                "packed_main": "ZmFrZS1wYWNrZWQ=",
                "scales": "ZmFrZS1zY2FsZXM=",
                "residual_mode": "qjl",
                "residual_data_keys": ["bits", "norms"],
                "d_head": 128,
                "d_rot": 128,
                "d_quant": 128,
                "algorithm": "turboquant_prod",
                "orig_dim": 128,
            }
        ],
    }
    return state, config


def test_state_schema_version_is_3() -> None:
    assert STATE_SCHEMA_VERSION == 3


def test_validate_state_accepts_canonical_v3_block_list() -> None:
    state, config = _canonical_v3_state()
    validate_state(state, config)


def test_validate_state_accepts_legacy_v2_payload() -> None:
    config = TurboQuantConfig()
    state = {
        "schema_version": 2,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        **_config_fields(config),
    }

    validate_state(state, config)


def test_validate_state_rejects_v3_without_blocks() -> None:
    state, config = _canonical_v3_state()
    state.pop("blocks")

    with pytest.raises(TurboQuantStateError, match="requires 'blocks'"):
        validate_state(state, config)


def test_validate_state_rejects_v3_empty_blocks_with_offset() -> None:
    state, config = _canonical_v3_state()
    state["blocks"] = []

    with pytest.raises(TurboQuantStateError, match="offset=4"):
        validate_state(state, config)


def test_validate_state_rejects_malformed_v3_block() -> None:
    state, config = _canonical_v3_state()
    state["blocks"] = [{"packed_main": 123}]

    with pytest.raises(TurboQuantStateError, match="Block 0"):
        validate_state(state, config)