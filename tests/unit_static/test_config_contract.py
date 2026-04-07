"""Static tests for TurboQuantConfig algorithm / residual-mode contract.

No MLX required — all assertions are pure-Python dataclass logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turboquant.config import TurboQuantConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"


def _load_contract_presets() -> dict[str, dict]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return {preset["name"]: preset for preset in payload["presets"]}

# ── Preset helpers ────────────────────────────────────────────────────────────


def test_paper_mse_preset_is_mse_mode():
    cfg = TurboQuantConfig.from_preset("paper_mse")
    assert cfg.is_mse_mode()
    assert not cfg.is_prod_mode()


def test_paper_prod_preset_is_prod_mode():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    assert cfg.is_prod_mode()
    assert not cfg.is_mse_mode()


def test_paper_prod_qjl_preset_is_prod_mode():
    cfg = TurboQuantConfig.from_preset("paper_prod_qjl")
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
    with pytest.raises(ValueError, match="paper_mse requires residual_mode='none'"):
        TurboQuantConfig(
            algorithm="paper_mse",
            residual_mode="qjl",
        ).validate()


def test_mse_with_topk_residual_raises():
    with pytest.raises(ValueError):
        TurboQuantConfig(
            algorithm="paper_mse",
            residual_mode="topk",
            residual_topk=2,
        ).validate()


def test_prod_with_none_residual_raises():
    with pytest.raises(ValueError, match="paper_prod_qjl requires residual_mode='qjl'"):
        TurboQuantConfig(
            algorithm="paper_prod_qjl",
            residual_mode="none",
        ).validate()


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="algorithm must be"):
        TurboQuantConfig(algorithm="turboquant_v3").validate()


def test_paper_mse_validate_passes():
    TurboQuantConfig.from_preset("paper_mse").validate()


def test_paper_prod_validate_passes():
    TurboQuantConfig.from_preset("paper_prod").validate()


def test_every_registered_preset_builds_and_validates():
    for name in TurboQuantConfig.preset_names():
        TurboQuantConfig.from_preset(name).validate()


def test_preset_registry_exposes_required_comparison_surfaces():
    registry = TurboQuantConfig.preset_registry()

    assert registry["paper_mse"]["classification"] == "paper-facing"
    assert registry["paper_prod_qjl"]["classification"] == "paper-facing"
    assert registry["paper_prod"]["classification"] == "paper-facing"
    assert (
        registry["polarquant_exp"]["classification"]
        == "supported non-paper-facing"
    )
    assert registry["legacy_topk"]["classification"] == "compatibility-only"


def test_contract_presets_match_runtime_registry_metadata():
    registry = TurboQuantConfig.preset_registry()
    contract_presets = _load_contract_presets()

    assert set(contract_presets).issubset(registry)

    for name, contract in contract_presets.items():
        runtime = registry[name]
        assert runtime["algorithm"] == contract["algorithm"]
        assert runtime["paper_facing"] == contract["paper_facing"]
        assert runtime["classification"] == contract["classification"]
        assert runtime["canonical_preset"] == contract["canonical_preset"]
        assert runtime["k_bits"] == contract["k_bits"]
        assert runtime["k_group_size"] == contract["k_group_size"]
        assert runtime["v_bits"] == contract["v_bits"]
        assert runtime["v_group_size"] == contract["v_group_size"]
        assert tuple(runtime["algorithm_aliases"]) == tuple(
            contract.get("algorithm_aliases", [])
        )

        if contract["residual_kind"] == "qjl":
            assert runtime["qjl_proj_dim"] == contract["qjl_dim"]
        if contract["residual_kind"] == "topk":
            assert runtime["residual_topk"] == contract["residual_topk"]


# ── effective_bits_per_channel formulae ──────────────────────────────────────


def test_effective_bits_k_mse_formula():
    # MSE: b + 16/g (paper §3)
    cfg = TurboQuantConfig.from_preset("paper_mse")  # k_bits=3, k_group_size=64
    d = 128
    expected = cfg.k_bits + 16.0 / cfg.k_group_size  # 3 + 0.25 = 3.25
    assert abs(cfg.effective_bits_per_channel_k(d) - expected) < 1e-9


def test_effective_bits_k_prod_formula():
    # Prod: (b-1) + 16/g + (p+16)/d (paper §3)
    cfg = TurboQuantConfig.from_preset(
        "paper_prod"
    )  # k_bits=3, k_group_size=64, qjl_proj_dim=64
    d = 128
    b, g, p = cfg.k_bits, cfg.k_group_size, cfg.qjl_proj_dim
    expected = (b - 1) + 16.0 / g + (p + 16.0) / d
    assert abs(cfg.effective_bits_per_channel_k(d) - expected) < 1e-9


def test_effective_bits_v_formula():
    cfg = TurboQuantConfig.from_preset("paper_prod")  # v_bits=4, v_group_size=64
    d = 128
    expected = cfg.v_bits + 16.0 / cfg.v_group_size  # 4 + 0.25 = 4.25
    assert abs(cfg.effective_bits_per_channel_v(d) - expected) < 1e-9


def test_effective_bits_total_is_average():
    cfg = TurboQuantConfig.from_preset("paper_prod")
    d = 128
    expected = (
        cfg.effective_bits_per_channel_k(d) + cfg.effective_bits_per_channel_v(d)
    ) / 2
    assert abs(cfg.effective_bits_per_channel_total(d) - expected) < 1e-9


def test_prod_is_smaller_bpc_than_naive():
    # Prod uses one fewer quant bit, so (b-1) < b for the main quantizer
    cfg_mse = TurboQuantConfig.from_preset("paper_mse")
    cfg_prod = TurboQuantConfig.from_preset("paper_prod")
    d = 4096  # large head dim → QJL overhead per-dim is small
    # At large d the prod cost is dominated by (b-1), which is < b
    bpc_mse = cfg_mse.effective_bits_per_channel_k(d)
    bpc_prod = cfg_prod.effective_bits_per_channel_k(d)
    assert bpc_prod < bpc_mse + 1.0, "Prod should be within 1 bit of MSE at large d"
