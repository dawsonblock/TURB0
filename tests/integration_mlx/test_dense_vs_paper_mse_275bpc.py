"""Integration test: dense vs paper_mse (~2.75 average KV bpc) perplexity gate.

Uses a reduced-bit ``paper_mse`` configuration
(``k_bits=2, k_group_size=64, v_bits=3, v_group_size=64``), which achieves
approximately 2.75 average KV bits per channel at ``d=128``.

With a real Llama model the quality gate is ``Δppl <= 1.5``, but that
real-model path is exploratory and opt-in. With the synthetic TinyModel
baseline (no download required) the gate is ``Δppl <= 100.0`` and only proves
that the pipeline runs end to end without numerical failure.

To test the exploratory real-model gate::

    export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
    export TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY=1
    python -m pytest tests/integration_mlx/test_dense_vs_paper_mse_275bpc.py -v
"""

from __future__ import annotations

import math
import os

import pytest

mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

_SAMPLE_PROMPTS = [
    (
        "The theory of relativity was developed by Albert Einstein "
        "in the early twentieth century."
    ),
    (
        "In machine learning, transformers have revolutionised natural "
        "language processing tasks."
    ),
    (
        "The Pythagorean theorem states that in a right triangle, "
        "the square of the hypotenuse"
    ),
]

_MAX_TOKENS = 64
_EXPECTED_AVG_BPC = 2.75
_MAX_DELTA_PPL_REAL = 1.5
_MAX_DELTA_PPL_SYNTHETIC = 100.0
_REAL_MODEL_OPT_IN_ENV = "TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY"


def _ppl(log_p, targets):
    import mlx.core as mx_

    token_count = targets.shape[0]
    nll = -float(mx_.sum(log_p[mx_.arange(token_count), targets]).item())
    return math.exp(nll / token_count)


def test_dense_vs_paper_mse_275bpc(tmp_path):
    """Dense vs paper_mse (~2.75 average KV bpc) perplexity comparison."""
    from mlx_lm.models.cache import make_prompt_cache
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list

    model_id = os.environ.get("TQ_TEST_LLAMA_MODEL", "")
    if model_id:
        if os.environ.get(_REAL_MODEL_OPT_IN_ENV) != "1":
            pytest.skip(
                "real-model Δppl gate is exploratory; set "
                f"{_REAL_MODEL_OPT_IN_ENV}=1 to opt in"
            )
        from mlx_lm import load as _load

        model, tokenizer = _load(model_id)
        max_delta_ppl = _MAX_DELTA_PPL_REAL
    else:
        from tests.helpers.tiny_model import TinyModel, TinyTokenizer

        model, tokenizer = TinyModel(), TinyTokenizer()
        mx.eval(model.parameters())
        max_delta_ppl = _MAX_DELTA_PPL_SYNTHETIC

    cfg = TurboQuantConfig.paper_mse(
        k_bits=2,
        k_group_size=64,
        v_bits=3,
        v_group_size=64,
        rotation="hadamard",
    )
    assert cfg.effective_bits_per_channel_total(128) == pytest.approx(
        _EXPECTED_AVG_BPC
    )

    delta_ppls = []
    for text in _SAMPLE_PROMPTS:
        tok_ids = tokenizer.encode(text)[: _MAX_TOKENS + 1]
        if len(tok_ids) < 4:
            continue
        input_ids = mx.array(tok_ids)[None]
        targets = input_ids[0, 1:]
        feed = input_ids[:, :-1]

        dense_cache = make_prompt_cache(model)
        dense_logits = model(feed, cache=dense_cache)[0]
        mx.eval(dense_logits)
        dense_log_p = dense_logits - mx.logsumexp(
            dense_logits, axis=-1, keepdims=True
        )
        dense_ppl = _ppl(dense_log_p, targets)

        tq_cache = make_prompt_cache(model)
        upgrade_cache_list(
            tq_cache, k_start=0, config=cfg, model_family="llama"
        )
        tq_logits = model(feed, cache=tq_cache)[0]
        mx.eval(tq_logits)
        tq_log_p = tq_logits - mx.logsumexp(
            tq_logits, axis=-1, keepdims=True
        )
        tq_ppl = _ppl(tq_log_p, targets)

        delta = tq_ppl - dense_ppl
        assert math.isfinite(delta), (
            f"Non-finite Δppl={delta} for prompt {text[:40]!r}"
        )
        delta_ppls.append(delta)

    assert delta_ppls, "No prompts produced valid results"
    mean_delta = sum(delta_ppls) / len(delta_ppls)
    assert mean_delta <= max_delta_ppl, (
        f"Mean Δppl={mean_delta:.4f} exceeds gate {max_delta_ppl} "
        f"at ~{_EXPECTED_AVG_BPC:.2f} average KV bpc (paper_mse)"
    )
