import pytest

from turboquant.config import TurboQuantConfig


def test_paper_mse_constructor_is_strict() -> None:
    cfg = TurboQuantConfig.paper_mse()
    assert cfg.algorithm_family() == "paper_mse"
    assert cfg.is_mse_mode()
    assert not cfg.is_prod_mode()
    cfg.validate()


def test_paper_prod_qjl_constructor_is_strict() -> None:
    cfg = TurboQuantConfig.paper_prod_qjl()
    assert cfg.algorithm_family() == "paper_prod_qjl"
    assert cfg.is_prod_mode()
    assert cfg.residual_mode == "qjl"
    cfg.validate()


def test_legacy_topk_constructor_is_strict() -> None:
    cfg = TurboQuantConfig.legacy_topk()
    assert cfg.algorithm_family() == "legacy_topk"
    assert cfg.residual_mode == "topk"
    cfg.validate()


def test_polarquant_constructor_requires_polar_quantizer() -> None:
    cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
    assert cfg.algorithm_family() == "polarquant_exp"
    assert cfg.quantizer_mode == "polar"
    cfg.validate()


def test_paper_prod_rejects_topk_residual() -> None:
    cfg = TurboQuantConfig(algorithm="paper_prod_qjl", residual_mode="topk")
    with pytest.raises(ValueError, match="paper_prod_qjl requires residual_mode='qjl'"):
        cfg.validate()


def test_legacy_aliases_normalize_to_canonical_algorithm_family() -> None:
    cfg = TurboQuantConfig(algorithm="turboquant_prod", residual_mode="qjl")
    assert cfg.algorithm_family() == "paper_prod_qjl"
    cfg.validate()