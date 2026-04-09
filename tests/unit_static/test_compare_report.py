"""
tests/unit_static/test_compare_report.py — ComparisonReport contract tests.

Covers the pure-Python parts of turboquant.eval.compare:

  * ComparisonReport instantiation and field access
  * auto-generated divergence_statement in __post_init__
  * custom divergence_statement bypasses auto-generation
  * to_dict structure and key presence
  * passed property (kl_bound_ok AND match_bound_ok)
  * DEFAULT_MAX_MEAN_KL and DEFAULT_MIN_MATCH_RATE constants

No MLX required.
"""

from __future__ import annotations

import pytest

from turboquant.eval.compare import (
    DEFAULT_MAX_MEAN_KL,
    DEFAULT_MIN_MATCH_RATE,
    ComparisonReport,
)


# ── Constants ──────────────────────────────────────────────────────────────────


def test_default_max_mean_kl_is_positive() -> None:
    assert DEFAULT_MAX_MEAN_KL > 0.0


def test_default_min_match_rate_in_unit_interval() -> None:
    assert 0.0 < DEFAULT_MIN_MATCH_RATE <= 1.0


# ── ComparisonReport instantiation ────────────────────────────────────────────


def test_comparison_report_stores_fields() -> None:
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
        model="llama",
        k_bits=3,
        v_bits=4,
    )
    assert report.mean_kl == 0.004
    assert report.max_kl == 0.021
    assert report.token_match_rate == 0.984
    assert report.n_tokens == 64
    assert report.kl_bound_ok is True
    assert report.match_bound_ok is True
    assert report.model == "llama"
    assert report.k_bits == 3
    assert report.v_bits == 4


# ── passed property ────────────────────────────────────────────────────────────


def test_passed_is_true_when_both_bounds_ok() -> None:
    report = ComparisonReport(
        mean_kl=0.01,
        max_kl=0.02,
        token_match_rate=0.99,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
    )
    assert report.passed is True


def test_passed_is_false_when_kl_bound_not_ok() -> None:
    report = ComparisonReport(
        mean_kl=0.5,
        max_kl=1.0,
        token_match_rate=0.99,
        n_tokens=64,
        kl_bound_ok=False,
        match_bound_ok=True,
    )
    assert report.passed is False


def test_passed_is_false_when_match_bound_not_ok() -> None:
    report = ComparisonReport(
        mean_kl=0.01,
        max_kl=0.02,
        token_match_rate=0.80,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=False,
    )
    assert report.passed is False


def test_passed_is_false_when_both_bounds_not_ok() -> None:
    report = ComparisonReport(
        mean_kl=0.5,
        max_kl=1.0,
        token_match_rate=0.80,
        n_tokens=64,
        kl_bound_ok=False,
        match_bound_ok=False,
    )
    assert report.passed is False


# ── divergence_statement auto-generation ─────────────────────────────────────


def test_divergence_statement_is_auto_generated_when_empty() -> None:
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
        k_bits=3,
        v_bits=4,
    )
    assert report.divergence_statement != ""
    # Should mention the mean_kl value
    assert "0.0040" in report.divergence_statement


def test_divergence_statement_includes_match_rate_percentage() -> None:
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
    )
    assert "98.4" in report.divergence_statement


def test_divergence_statement_includes_k_bits_and_v_bits() -> None:
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
        k_bits=3,
        v_bits=4,
    )
    assert "3" in report.divergence_statement
    assert "4" in report.divergence_statement


def test_custom_divergence_statement_is_preserved() -> None:
    custom = "Custom statement about compression quality."
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
        divergence_statement=custom,
    )
    assert report.divergence_statement == custom


# ── to_dict ───────────────────────────────────────────────────────────────────


def test_to_dict_contains_required_keys() -> None:
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=True,
    )
    d = report.to_dict()
    required = {
        "mean_kl", "max_kl", "token_match_rate", "n_tokens",
        "kl_bound_ok", "match_bound_ok", "model", "k_bits", "v_bits",
        "divergence_statement",
    }
    assert required.issubset(d.keys())


def test_to_dict_values_match_fields() -> None:
    report = ComparisonReport(
        mean_kl=0.004,
        max_kl=0.021,
        token_match_rate=0.984,
        n_tokens=64,
        kl_bound_ok=True,
        match_bound_ok=False,
        model="gemma",
        k_bits=3,
        v_bits=4,
    )
    d = report.to_dict()
    assert d["mean_kl"] == 0.004
    assert d["max_kl"] == 0.021
    assert d["token_match_rate"] == 0.984
    assert d["n_tokens"] == 64
    assert d["kl_bound_ok"] is True
    assert d["match_bound_ok"] is False
    assert d["model"] == "gemma"
    assert d["k_bits"] == 3
    assert d["v_bits"] == 4


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_comparison_report_defaults_model_to_unknown() -> None:
    report = ComparisonReport(
        mean_kl=0.0,
        max_kl=0.0,
        token_match_rate=1.0,
        n_tokens=10,
        kl_bound_ok=True,
        match_bound_ok=True,
    )
    assert report.model == "unknown"


def test_comparison_report_perfect_match_passes() -> None:
    report = ComparisonReport(
        mean_kl=0.0,
        max_kl=0.0,
        token_match_rate=1.0,
        n_tokens=100,
        kl_bound_ok=True,
        match_bound_ok=True,
    )
    assert report.passed is True


def test_comparison_report_zero_token_match_fails() -> None:
    report = ComparisonReport(
        mean_kl=2.0,
        max_kl=5.0,
        token_match_rate=0.0,
        n_tokens=100,
        kl_bound_ok=False,
        match_bound_ok=False,
    )
    assert report.passed is False
