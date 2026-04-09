"""
tests/unit_static/test_state_extended.py — extended validate_state tests.

Covers paths not exercised by test_state_contract.py:

  * Missing 'schema_version' key
  * Non-integer schema_version
  * Unsupported schema version (99)
  * Missing required scalar keys (offset, d_head, etc.)
  * Negative / non-integer offset
  * v1 state: offset > 0 with missing k_packed → corrupt
  * v2 state: missing config keys
  * v4 metadata: invalid residual_kind, rotation_type, algorithm
  * v4 metadata: polarquant_exp with identity rotation
  * v4 metadata: paper_mse with non-none residual_kind
  * v4 metadata: legacy_topk with wrong residual_kind
  * Config mismatch detection (k_bits, rotation, algorithm)
  * Block payload: non-list blocks, empty blocks with offset>0, bad block

No MLX required.
"""

from __future__ import annotations

import pytest

from turboquant.config import TurboQuantConfig
from turboquant.errors import TurboQuantStateError
from turboquant.runtime.state import STATE_SCHEMA_VERSION, validate_state


def _minimal_v4_state(config: TurboQuantConfig | None = None) -> dict:
    """Return a minimal valid v4 state dict for paper_prod_qjl."""
    cfg = config or TurboQuantConfig.paper_prod_qjl()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        "k_bits": cfg.k_bits,
        "k_group_size": cfg.k_group_size,
        "v_bits": cfg.v_bits,
        "v_group_size": cfg.v_group_size,
        "v_enabled": cfg.v_enabled,
        "rotation": cfg.rotation,
        "rotation_seed": cfg.rotation_seed,
        "residual_topk": cfg.residual_topk,
        "scale_dtype": cfg.scale_dtype,
        "v_scale_dtype": cfg.v_scale_dtype,
        "eps": cfg.eps,
        "algorithm": "paper_prod_qjl",
        "rotation_type": "hadamard",
        "residual_kind": "qjl",
        "qjl_dim": 64,
        "qjl_seed": 42,
        "codebook_id": "lloydmax-qjl-k2-m64-hadamard",
        "main_bits": 2,
        "residual_bits": 1,
        "blocks": [],
    }


# ── Missing / invalid schema_version ──────────────────────────────────────────


def test_validate_state_rejects_missing_schema_version() -> None:
    with pytest.raises(TurboQuantStateError, match="schema_version"):
        validate_state({"offset": 0, "d_head": 128, "d_pad": 128, "v_dim": 0, "v_pad": 0})


def test_validate_state_rejects_non_integer_schema_version() -> None:
    with pytest.raises(TurboQuantStateError, match="schema_version"):
        validate_state({"schema_version": "4", "offset": 0, "d_head": 0, "d_pad": 0, "v_dim": 0, "v_pad": 0})


def test_validate_state_rejects_unsupported_version() -> None:
    with pytest.raises(TurboQuantStateError, match="incompatible"):
        validate_state({"schema_version": 99, "offset": 0, "d_head": 0, "d_pad": 0, "v_dim": 0, "v_pad": 0})


def test_validate_state_rejects_version_zero() -> None:
    with pytest.raises(TurboQuantStateError):
        validate_state({"schema_version": 0, "offset": 0, "d_head": 0, "d_pad": 0, "v_dim": 0, "v_pad": 0})


# ── Missing required scalar keys ──────────────────────────────────────────────


def test_validate_state_rejects_missing_offset() -> None:
    state = _minimal_v4_state()
    state.pop("offset")
    with pytest.raises(TurboQuantStateError, match="missing required keys"):
        validate_state(state)


def test_validate_state_rejects_missing_d_head() -> None:
    state = _minimal_v4_state()
    state.pop("d_head")
    with pytest.raises(TurboQuantStateError, match="missing required keys"):
        validate_state(state)


def test_validate_state_rejects_negative_offset() -> None:
    state = _minimal_v4_state()
    state["offset"] = -1
    with pytest.raises(TurboQuantStateError, match="offset"):
        validate_state(state)


def test_validate_state_rejects_non_integer_offset() -> None:
    state = _minimal_v4_state()
    state["offset"] = "four"
    with pytest.raises(TurboQuantStateError, match="offset"):
        validate_state(state)


# ── v4 metadata validation ────────────────────────────────────────────────────


def test_validate_state_rejects_unsupported_algorithm_in_v4_metadata() -> None:
    state = _minimal_v4_state()
    state["algorithm"] = "super_quant_v9"
    with pytest.raises(TurboQuantStateError, match="unsupported algorithm"):
        validate_state(state)


def test_validate_state_rejects_empty_codebook_id() -> None:
    state = _minimal_v4_state()
    state["codebook_id"] = ""
    with pytest.raises(TurboQuantStateError, match="codebook_id"):
        validate_state(state)


def test_validate_state_rejects_empty_rotation_type() -> None:
    state = _minimal_v4_state()
    state["rotation_type"] = ""
    with pytest.raises(TurboQuantStateError, match="rotation_type"):
        validate_state(state)


def test_validate_state_rejects_unsupported_residual_kind() -> None:
    state = _minimal_v4_state()
    state["residual_kind"] = "sparse"
    with pytest.raises(TurboQuantStateError, match="residual_kind"):
        validate_state(state)


def test_validate_state_rejects_paper_mse_with_qjl_residual_kind() -> None:
    cfg = TurboQuantConfig.paper_mse()
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        "k_bits": cfg.k_bits,
        "k_group_size": cfg.k_group_size,
        "v_bits": cfg.v_bits,
        "v_group_size": cfg.v_group_size,
        "v_enabled": cfg.v_enabled,
        "rotation": cfg.rotation,
        "rotation_seed": cfg.rotation_seed,
        "residual_topk": cfg.residual_topk,
        "scale_dtype": cfg.scale_dtype,
        "v_scale_dtype": cfg.v_scale_dtype,
        "eps": cfg.eps,
        "algorithm": "paper_mse",
        "rotation_type": "hadamard",
        "residual_kind": "qjl",  # wrong for paper_mse
        "qjl_dim": 64,
        "qjl_seed": 42,
        "codebook_id": "test",
        "main_bits": 3,
        "residual_bits": 1,
        "blocks": [],
    }
    with pytest.raises(TurboQuantStateError, match="paper_mse"):
        validate_state(state)


def test_validate_state_rejects_legacy_topk_with_wrong_residual_kind() -> None:
    state = _minimal_v4_state()
    state["algorithm"] = "legacy_topk"
    state["residual_kind"] = "qjl"
    with pytest.raises(TurboQuantStateError, match="legacy_topk"):
        validate_state(state)


def test_validate_state_rejects_polarquant_with_identity_rotation() -> None:
    state = _minimal_v4_state()
    state["algorithm"] = "polarquant_exp"
    state["rotation_type"] = "identity"
    state["residual_kind"] = "none"
    state["qjl_dim"] = 0
    state["residual_bits"] = 0
    # Must fail on rotation check for polarquant_exp
    with pytest.raises(TurboQuantStateError, match="randomized preconditioning"):
        validate_state(state)


def test_validate_state_rejects_negative_qjl_dim() -> None:
    state = _minimal_v4_state()
    state["qjl_dim"] = -1
    with pytest.raises(TurboQuantStateError, match="qjl_dim"):
        validate_state(state)


def test_validate_state_rejects_qjl_residual_bits_not_one() -> None:
    state = _minimal_v4_state()
    state["residual_bits"] = 2
    with pytest.raises(TurboQuantStateError, match="residual_bits"):
        validate_state(state)


# ── v1 state ──────────────────────────────────────────────────────────────────


def test_validate_state_accepts_v1_state_with_zero_offset() -> None:
    state = {
        "schema_version": 1,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
    }
    validate_state(state)


def test_validate_state_rejects_v1_with_nonzero_offset_and_no_k_packed() -> None:
    state = {
        "schema_version": 1,
        "offset": 64,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        "k_packed": None,
    }
    with pytest.raises(TurboQuantStateError, match="k_packed"):
        validate_state(state)


# ── v2 state ──────────────────────────────────────────────────────────────────


def test_validate_state_accepts_v2_state_with_zero_offset() -> None:
    state = {
        "schema_version": 2,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        # required v2 config keys
        "k_bits": 3,
        "k_group_size": 64,
        "v_bits": 4,
        "v_group_size": 64,
        "v_enabled": True,
        "rotation": "hadamard",
        "rotation_seed": 1337,
        "residual_topk": 0,
        "scale_dtype": "float16",
        "v_scale_dtype": "float16",
        "eps": 1e-6,
    }
    validate_state(state)


def test_validate_state_rejects_v2_state_with_missing_config_keys() -> None:
    state = {
        "schema_version": 2,
        "offset": 0,
        "d_head": 128,
        "d_pad": 128,
        "v_dim": 0,
        "v_pad": 0,
        # missing several config keys
    }
    with pytest.raises(TurboQuantStateError, match="missing config keys"):
        validate_state(state)


# ── v3 block payload ──────────────────────────────────────────────────────────


def test_validate_state_rejects_v3_with_non_list_blocks() -> None:
    state = _minimal_v4_state()
    state["schema_version"] = 3
    state["blocks"] = {}  # dict instead of list
    with pytest.raises(TurboQuantStateError, match="list"):
        validate_state(state)


def test_validate_state_rejects_v3_with_nonzero_offset_and_empty_blocks() -> None:
    state = _minimal_v4_state()
    state["schema_version"] = 3
    state["offset"] = 64
    state["blocks"] = []
    with pytest.raises(TurboQuantStateError, match="blocks.*empty"):
        validate_state(state)


def test_validate_state_rejects_block_missing_required_keys() -> None:
    state = _minimal_v4_state()
    state["offset"] = 4
    state["blocks"] = [
        {
            # missing most keys
            "packed_main": None,
            "scales": None,
        }
    ]
    with pytest.raises(TurboQuantStateError, match="missing required keys"):
        validate_state(state)


# ── Config mismatch detection ─────────────────────────────────────────────────


def test_validate_state_rejects_k_bits_mismatch() -> None:
    cfg = TurboQuantConfig.paper_prod_qjl()
    state = _minimal_v4_state(cfg)
    state["k_bits"] = 99  # wrong
    with pytest.raises(TurboQuantStateError, match="k_bits"):
        validate_state(state, cfg)


def test_validate_state_rejects_rotation_mismatch() -> None:
    cfg = TurboQuantConfig.paper_prod_qjl()
    state = _minimal_v4_state(cfg)
    state["rotation"] = "identity"
    state["rotation_type"] = "identity"
    with pytest.raises(TurboQuantStateError, match="rotation"):
        validate_state(state, cfg)


def test_validate_state_rejects_algorithm_mismatch() -> None:
    cfg = TurboQuantConfig.paper_prod_qjl()
    state = _minimal_v4_state(cfg)
    state["algorithm"] = "paper_mse"
    state["residual_kind"] = "none"
    state["qjl_dim"] = 0
    state["residual_bits"] = 0
    with pytest.raises(TurboQuantStateError):
        validate_state(state, cfg)


def test_validate_state_no_config_skips_mismatch_check() -> None:
    state = _minimal_v4_state()
    # Should not raise even though no config is provided.
    validate_state(state, config=None)
