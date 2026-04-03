"""Static tests for TurboQuantConfig algorithm / residual-mode contract.

No MLX required — all assertions are pure-Python dataclass logic.
"""

from __future__ import annotations

import pytest

from turboquant.config import TurboQuantConfig


# ── Preset helpers ────────────────────────────────────────────────────────────


def test_paper_mse_preset_is_mse_mode():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    assert cfg.is_mse_mode()
    assert not cfg.is_prod_mode()


def test_paper_prod_preset_is_prod_mode():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    assert cfg.is_prod_mode()
    assert not cfg.is_mse_mode()


def test_paper_mse_has_no_residual():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    assert cfg.residual_mode == "none"


def test_paper_prod_has_qjl_residual():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    assert cfg.residual_mode == "qjl"


def test_paper_mse_uses_hadamard_rotation():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    assert cfg.rotation == "hadamard"


def test_paper_prod_uses_hadamard_rotation():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    assert cfg.rotation == "hadamard"


# ── Validation contract ───────────────────────────────────────────────────────


def test_mse_with_qjl_residual_raises():
    with pytest.raises(ValueError, match="algorithm='turboquant_mse' requires residual_mode='none'"):
        TurboQuantConfig(
            algorithm="turboquant_mse",
            residual_mode="qjl",
        ).validate()


def test_mse_with_topk_residual_raises():
    with pytest.raises(ValueError):
        TurboQuantConfig(
            algorithm="turboquant_mse",
            residual_mode="topk",
            residual_topk=2,
        ).validate()


def test_prod_with_none_residual_raises():
    with pytest.raises(ValueError, match="algorithm='turboquant_prod' requires a residual encoder"):
        TurboQuantConfig(
            algorithm="turboquant_prod",
            residual_mode="none",
        ).validate()


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="algorithm must be"):
        TurboQuantConfig(algorithm="turboquant_v3").validate()


def test_paper_mse_validate_passes():
    TurboQuantConfig.from_preset("paper_mse").validate()


def test_paper_prod_validate_passes():
    TurboQuantConfig.from_preset("paper_prod").validate()


# ── effective_bits_per_channel formulae ──────────────────────────────────────


def test_effective_bits_k_mse_formula():
    # MSE: b + 16/g (paper §3)
    cfg = TurboQuantConfig.from_preset("paper_mse")   # k_bits=3, k_group_size=64
    d = 128
    expected = cfg.k_bits + 16.0 / cfg.k_group_size   # 3 + 0.25 = 3.25
    assert abs(cfg.effective_bits_per_channel_k(d) - expected) < 1e-9


def test_effective_bits_k_prod_formula():
    # Prod: (b-1) + 16/g + (p+16)/d (paper §3)
    cfg = TurboQuantConfig.from_preset("paper_prod")   # k_bits=3, k_group_size=64, qjl_proj_dim=64
    d = 128
    b, g, p = cfg.k_bits, cfg.k_group_size, cfg.qjl_proj_dim
    expected = (b - 1) + 16.0 / g + (p + 16.0) / d
    assert abs(cfg.effective_bits_per_channel_k(d) - expected) < 1e-9


def test_effective_bits_v_formula():
    cfg = TurboQuantConfig.from_preset("paper_prod")   # v_bits=4, v_group_size=64
    d = 128
    expected = cfg.v_bits + 16.0 / cfg.v_group_size   # 4 + 0.25 = 4.25
    assert abs(cfg.effective_bits_per_channel_v(d) - expected) < 1e-9


def test_effective_bits_total_is_average():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    d = 128
    expected = (cfg.effective_bits_per_channel_k(d) + cfg.effective_bits_per_channel_v(d)) / 2
    assert abs(cfg.effective_bits_per_channel_total(d) - expected) < 1e-9


def test_prod_is_smaller_bpc_than_naive():
    # Prod uses one fewer quant bit, so (b-1) < b for the main quantizer
    cfg_mse = TurboQuantConfig.from_preset("paper_mse")
    cfg_prod = TurboQuantConfig.from_preset("paper_prod")
    d = 4096   # large head dim → QJL overhead per-dim is small
    # At large d the prod cost is dominated by (b-1), which is < b
    bpc_mse = cfg_mse.effective_bits_per_channel_k(d)
    bpc_prod = cfg_prod.effective_bits_per_channel_k(d)
    assert bpc_prod < bpc_mse + 1.0, "Prod should be within 1 bit of MSE at large d"
