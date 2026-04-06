from typing import Any, cast

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


def _canonical_v4_state() -> tuple[dict[str, object], TurboQuantConfig]:
    config = TurboQuantConfig.paper_prod_qjl()
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "offset": 4,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        **_config_fields(config),
        "algorithm": "paper_prod_qjl",
        "rotation_type": "hadamard",
        "residual_kind": "qjl",
        "qjl_dim": 64,
        "qjl_seed": 42,
        "codebook_id": "lloydmax-qjl-k2-m64-hadamard",
        "main_bits": 2,
        "residual_bits": 1,
        "blocks": [
            {
                "packed_main": "ZmFrZS1wYWNrZWQ=",
                "scales": "ZmFrZS1zY2FsZXM=",
                "residual_mode": "qjl",
                "residual_data_keys": ["bits", "norms"],
                "d_head": 128,
                "d_rot": 128,
                "d_quant": 128,
                "algorithm": "paper_prod_qjl",
                "orig_dim": 128,
            }
        ],
    }
    return state, config


def _polar_v4_state() -> tuple[dict[str, object], TurboQuantConfig]:
    config = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "offset": 4,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        **_config_fields(config),
        "algorithm": "polarquant_exp",
        "rotation_type": "random_orthogonal",
        "residual_kind": "none",
        "qjl_dim": 0,
        "qjl_seed": 0,
        "codebook_id": "polar-angle-codebook-exp",
        "main_bits": 3,
        "residual_bits": 0,
        "blocks": [
            {
                "packed_main": None,
                "scales": None,
                "residual_mode": "none",
                "residual_data_keys": [],
                "d_head": 128,
                "d_rot": 128,
                "d_quant": 128,
                "algorithm": "polarquant_exp",
                "orig_dim": 128,
                "polar_payload": {
                    "angle_codes": ["ZmFrZS1jb2RlLTE=", "ZmFrZS1jb2RlLTI="],
                    "final_radii": "ZmFrZS1yYWRpaQ==",
                    "d_orig": 128,
                    "d_pad": 128,
                    "n_levels": 2,
                },
            }
        ],
    }
    return state, config


def test_state_schema_version_is_4() -> None:
    assert STATE_SCHEMA_VERSION == 4


def test_validate_state_accepts_canonical_v4_block_list() -> None:
    state, config = _canonical_v4_state()
    validate_state(state, config)


def test_validate_state_accepts_legacy_v3_payload() -> None:
    config = TurboQuantConfig.paper_prod_qjl()
    state = {
        "schema_version": 3,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        **_config_fields(config),
        "blocks": [],
    }

    validate_state(state, config)


def test_validate_state_rejects_v4_without_algorithm_metadata() -> None:
    state, config = _canonical_v4_state()
    state.pop("algorithm")

    with pytest.raises(TurboQuantStateError, match="missing metadata keys"):
        validate_state(state, config)


def test_validate_state_rejects_v4_without_qjl_metadata() -> None:
    state, config = _canonical_v4_state()
    state["qjl_dim"] = 0

    with pytest.raises(TurboQuantStateError, match="qjl_dim > 0"):
        validate_state(state, config)


def test_validate_state_rejects_v4_paper_prod_without_qjl_kind() -> None:
    state, config = _canonical_v4_state()
    state["residual_kind"] = "none"

    with pytest.raises(TurboQuantStateError, match="paper_prod_qjl"):
        validate_state(state, config)


def test_validate_state_rejects_v4_without_blocks() -> None:
    state, config = _canonical_v4_state()
    state.pop("blocks")

    with pytest.raises(TurboQuantStateError, match="requires 'blocks'"):
        validate_state(state, config)


def test_validate_state_accepts_polar_v4_block_list() -> None:
    state, config = _polar_v4_state()
    validate_state(state, config)


def test_validate_state_rejects_polar_v4_without_payload() -> None:
    state, config = _polar_v4_state()
    blocks = cast(list[dict[str, Any]], state["blocks"])
    blocks[0].pop("polar_payload")

    with pytest.raises(TurboQuantStateError, match="must include 'polar_payload'"):
        validate_state(state, config)
