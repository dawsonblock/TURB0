"""
tests/unit_static/test_config_extended.py — extended TurboQuantConfig tests.

Covers parts of config.py not already tested by test_config_contract.py
and test_algorithm_modes.py:

  * validate() — full boundary / edge-case coverage for every guard
  * to_state_dict() — keys present, values match live config fields
  * from_legacy_kwargs() — legacy key mapping, residual-mode inference, errors
  * normalize_algorithm() — all registered aliases
  * preset_metadata() — unknown preset error path

No MLX required.
"""

from __future__ import annotations

import pytest

from turboquant.config import TurboQuantConfig


# ── validate() — k_bits bounds ────────────────────────────────────────────────


def test_validate_rejects_zero_k_bits() -> None:
    with pytest.raises(ValueError, match="k_bits"):
        TurboQuantConfig.paper_mse(k_bits=0).validate()


def test_validate_rejects_negative_k_bits() -> None:
    with pytest.raises(ValueError, match="k_bits"):
        TurboQuantConfig.paper_mse(k_bits=-1).validate()


def test_validate_rejects_k_bits_above_8() -> None:
    with pytest.raises(ValueError, match="k_bits"):
        TurboQuantConfig.paper_mse(k_bits=9).validate()


def test_validate_accepts_k_bits_1() -> None:
    TurboQuantConfig.paper_mse(k_bits=1).validate()


def test_validate_accepts_k_bits_8() -> None:
    TurboQuantConfig.paper_mse(k_bits=8).validate()


# ── validate() — k_group_size bounds ─────────────────────────────────────────


def test_validate_rejects_zero_k_group_size() -> None:
    with pytest.raises(ValueError, match="k_group_size"):
        TurboQuantConfig.paper_mse(k_group_size=0).validate()


def test_validate_rejects_negative_k_group_size() -> None:
    with pytest.raises(ValueError, match="k_group_size"):
        TurboQuantConfig.paper_mse(k_group_size=-4).validate()


def test_validate_accepts_positive_k_group_size() -> None:
    TurboQuantConfig.paper_mse(k_group_size=32).validate()


# ── validate() — v_bits bounds ────────────────────────────────────────────────


def test_validate_rejects_zero_v_bits_when_v_enabled() -> None:
    with pytest.raises(ValueError, match="v_bits"):
        TurboQuantConfig.paper_mse(v_bits=0, v_enabled=True).validate()


def test_validate_rejects_v_bits_above_8_when_v_enabled() -> None:
    with pytest.raises(ValueError, match="v_bits"):
        TurboQuantConfig.paper_mse(v_bits=9, v_enabled=True).validate()


def test_validate_accepts_v_bits_1_when_v_enabled() -> None:
    TurboQuantConfig.paper_mse(v_bits=1, v_enabled=True).validate()


def test_validate_skips_v_bits_check_when_v_disabled() -> None:
    # v_bits=0 is irrelevant when v_enabled=False
    cfg = TurboQuantConfig(
        algorithm="paper_mse",
        quantizer_mode="scalar",
        residual_mode="none",
        v_bits=0,
        v_enabled=False,
    )
    cfg.validate()


# ── validate() — v_group_size bounds ─────────────────────────────────────────


def test_validate_rejects_zero_v_group_size_when_v_enabled() -> None:
    with pytest.raises(ValueError, match="v_group_size"):
        TurboQuantConfig.paper_mse(v_group_size=0, v_enabled=True).validate()


# ── validate() — rotation values ─────────────────────────────────────────────


def test_validate_rejects_unknown_rotation() -> None:
    with pytest.raises(ValueError, match="Unsupported rotation"):
        TurboQuantConfig.paper_mse(rotation="random_rotation_xyz").validate()


def test_validate_accepts_hadamard_rotation() -> None:
    TurboQuantConfig.paper_mse(rotation="hadamard").validate()


def test_validate_accepts_identity_rotation() -> None:
    TurboQuantConfig.paper_mse(rotation="identity").validate()


def test_validate_accepts_random_orthogonal_rotation() -> None:
    TurboQuantConfig.polarquant_exp(rotation="random_orthogonal").validate()


# ── validate() — residual_mode values ────────────────────────────────────────


def test_validate_rejects_unknown_residual_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported residual_mode"):
        TurboQuantConfig(
            algorithm="paper_mse",
            quantizer_mode="scalar",
            residual_mode="sparse_topk_extended",
        ).validate()


# ── validate() — quantizer_mode values ───────────────────────────────────────


def test_validate_rejects_unknown_quantizer_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported quantizer_mode"):
        TurboQuantConfig(
            algorithm="paper_mse",
            residual_mode="none",
            quantizer_mode="vector",
        ).validate()


# ── validate() — paper_mse cross-constraints ─────────────────────────────────


def test_validate_paper_mse_rejects_polar_quantizer_mode() -> None:
    with pytest.raises(ValueError, match="paper_mse requires quantizer_mode"):
        TurboQuantConfig(
            algorithm="paper_mse",
            residual_mode="none",
            quantizer_mode="polar",
        ).validate()


# ── validate() — paper_prod_qjl cross-constraints ────────────────────────────


def test_validate_paper_prod_rejects_polar_quantizer_mode() -> None:
    with pytest.raises(ValueError, match="paper_prod_qjl requires quantizer_mode"):
        TurboQuantConfig(
            algorithm="paper_prod_qjl",
            residual_mode="qjl",
            quantizer_mode="polar",
        ).validate()


# ── validate() — legacy_topk cross-constraints ───────────────────────────────


def test_validate_legacy_topk_rejects_qjl_residual() -> None:
    with pytest.raises(ValueError, match="legacy_topk requires residual_mode='topk'"):
        TurboQuantConfig(
            algorithm="legacy_topk",
            quantizer_mode="scalar",
            residual_mode="qjl",
            residual_topk=2,
        ).validate()


def test_validate_legacy_topk_rejects_polar_quantizer() -> None:
    with pytest.raises(ValueError, match="legacy_topk requires quantizer_mode"):
        TurboQuantConfig(
            algorithm="legacy_topk",
            quantizer_mode="polar",
            residual_mode="topk",
            residual_topk=2,
        ).validate()


# ── validate() — polarquant_exp cross-constraints ────────────────────────────


def test_validate_polarquant_rejects_identity_rotation() -> None:
    with pytest.raises(ValueError, match="randomized preconditioning"):
        TurboQuantConfig(
            algorithm="polarquant_exp",
            quantizer_mode="polar",
            residual_mode="none",
            rotation="identity",
        ).validate()


def test_validate_polarquant_rejects_scalar_quantizer() -> None:
    with pytest.raises(ValueError, match="polarquant_exp requires quantizer_mode"):
        TurboQuantConfig(
            algorithm="polarquant_exp",
            quantizer_mode="scalar",
            residual_mode="none",
            rotation="random_orthogonal",
        ).validate()


# ── validate() — topk residual constraints ───────────────────────────────────


def test_validate_topk_residual_requires_positive_residual_topk() -> None:
    with pytest.raises(ValueError, match="residual_topk must be > 0"):
        TurboQuantConfig(
            algorithm="legacy_topk",
            quantizer_mode="scalar",
            residual_mode="topk",
            residual_topk=0,
        ).validate()


# ── validate() — qjl residual constraints ────────────────────────────────────


def test_validate_qjl_residual_rejects_non_1_qjl_bits() -> None:
    with pytest.raises(ValueError, match="Only 1-bit QJL"):
        TurboQuantConfig(
            algorithm="paper_prod_qjl",
            quantizer_mode="scalar",
            residual_mode="qjl",
            qjl_bits=2,
        ).validate()


def test_validate_qjl_residual_rejects_zero_proj_dim() -> None:
    with pytest.raises(ValueError, match="qjl_proj_dim must be > 0"):
        TurboQuantConfig(
            algorithm="paper_prod_qjl",
            quantizer_mode="scalar",
            residual_mode="qjl",
            qjl_bits=1,
            qjl_proj_dim=0,
        ).validate()


# ── normalize_algorithm ───────────────────────────────────────────────────────


def test_normalize_algorithm_identity_for_canonical_name() -> None:
    assert TurboQuantConfig.normalize_algorithm("paper_mse") == "paper_mse"
    assert TurboQuantConfig.normalize_algorithm("paper_prod_qjl") == "paper_prod_qjl"
    assert TurboQuantConfig.normalize_algorithm("legacy_topk") == "legacy_topk"
    assert TurboQuantConfig.normalize_algorithm("polarquant_exp") == "polarquant_exp"


def test_normalize_algorithm_turboquant_mse_alias() -> None:
    assert TurboQuantConfig.normalize_algorithm("turboquant_mse") == "paper_mse"


def test_normalize_algorithm_turboquant_prod_alias() -> None:
    assert TurboQuantConfig.normalize_algorithm("turboquant_prod") == "paper_prod_qjl"


def test_normalize_algorithm_paper_prod_alias() -> None:
    assert TurboQuantConfig.normalize_algorithm("paper_prod") == "paper_prod_qjl"


def test_normalize_algorithm_unknown_is_returned_unchanged() -> None:
    assert TurboQuantConfig.normalize_algorithm("my_custom_algo") == "my_custom_algo"


# ── to_state_dict ─────────────────────────────────────────────────────────────


def test_to_state_dict_contains_required_keys() -> None:
    cfg = TurboQuantConfig.paper_prod_qjl()
    sd = cfg.to_state_dict()
    required = {
        "k_bits", "k_group_size", "v_bits", "v_group_size", "v_enabled",
        "rotation", "rotation_seed", "rotation_pad_to_pow2",
        "residual_mode", "residual_topk", "resid_scale_bits",
        "scale_dtype", "v_scale_dtype", "eps", "block_tokens",
        "qjl_proj_dim", "qjl_seed", "qjl_bits",
        "quantizer_mode", "algorithm", "return_mode",
    }
    assert required.issubset(sd.keys())


def test_to_state_dict_reflects_config_values() -> None:
    cfg = TurboQuantConfig.paper_mse(k_bits=4, k_group_size=32)
    sd = cfg.to_state_dict()
    assert sd["k_bits"] == 4
    assert sd["k_group_size"] == 32
    assert sd["algorithm"] == "paper_mse"
    assert sd["residual_mode"] == "none"
    assert sd["quantizer_mode"] == "scalar"


def test_to_state_dict_algorithm_is_normalised() -> None:
    # alias 'turboquant_mse' should be stored as canonical 'paper_mse'
    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        quantizer_mode="scalar",
        residual_mode="none",
    )
    sd = cfg.to_state_dict()
    assert sd["algorithm"] == "paper_mse"


# ── preset_metadata error path ────────────────────────────────────────────────


def test_preset_metadata_raises_for_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        TurboQuantConfig.preset_metadata("nonexistent_preset")


def test_from_preset_raises_for_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        TurboQuantConfig.from_preset("nonexistent_preset")


# ── from_legacy_kwargs ────────────────────────────────────────────────────────


def test_from_legacy_kwargs_default_is_paper_prod_qjl() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs()
    assert cfg.algorithm_family() == "paper_prod_qjl"
    assert cfg.residual_mode == "qjl"


def test_from_legacy_kwargs_none_residual_mode_yields_paper_mse() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs(residual_mode="none")
    assert cfg.algorithm_family() == "paper_mse"
    assert cfg.residual_mode == "none"


def test_from_legacy_kwargs_qjl_residual_yields_paper_prod_qjl() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs(residual_mode="qjl")
    assert cfg.algorithm_family() == "paper_prod_qjl"
    assert cfg.residual_mode == "qjl"


def test_from_legacy_kwargs_topk_residual_requires_explicit_algorithm() -> None:
    with pytest.raises(ValueError, match="algorithm='legacy_topk'"):
        TurboQuantConfig.from_legacy_kwargs(
            residual_mode="topk",
            residual_topk=2,
        )


def test_from_legacy_kwargs_topk_with_explicit_algorithm() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs(
        algorithm="legacy_topk",
        residual_mode="topk",
        residual_topk=3,
    )
    assert cfg.algorithm_family() == "legacy_topk"
    assert cfg.residual_topk == 3


def test_from_legacy_kwargs_respects_k_bits() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs(k_bits=4)
    assert cfg.k_bits == 4


def test_from_legacy_kwargs_rotation_mode_alias() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs(rotation_mode="hadamard")
    assert cfg.rotation == "hadamard"


def test_from_legacy_kwargs_group_size_alias() -> None:
    cfg = TurboQuantConfig.from_legacy_kwargs(group_size=32)
    assert cfg.k_group_size == 32


def test_from_legacy_kwargs_validates_result() -> None:
    # Invalid algorithm must raise ValueError via validate()
    with pytest.raises(ValueError):
        TurboQuantConfig.from_legacy_kwargs(algorithm="bogus_algo")
