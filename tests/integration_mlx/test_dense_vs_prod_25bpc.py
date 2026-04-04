"""Integration test: dense vs TurboQuant (~2.5 bpc) perplexity gate.

Uses a reduced-bit configuration (k_bits=2, k_group_size=64, v_bits=3,
v_group_size=64, algorithm="turboquant_mse", residual_mode="none") which
achieves approximately 2.5 bpc average.

With a real Llama model the quality gate is Δppl ≤ 1.5 (larger tolerance
than 3.5 bpc because 2.5 bpc is a more aggressive compression), but that
real-model path is exploratory and opt-in.
With the synthetic tiny model (no download required) the gate is Δppl ≤ 100.0
— this only verifies that the pipeline runs end-to-end without errors.

To test the exploratory real-model gate::

    export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
    export TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY=1
    python -m pytest tests/integration_mlx/test_dense_vs_prod_25bpc.py -v
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
# Quality gate: tight for real models, loose for synthetic model.
_MAX_DELTA_PPL_REAL = 1.5
_MAX_DELTA_PPL_SYNTHETIC = 100.0
_REAL_MODEL_OPT_IN_ENV = "TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY"


def _ppl(log_p, targets):
    import mlx.core as mx_
    T = targets.shape[0]
    nll = -float(mx_.sum(log_p[mx_.arange(T), targets]).item())
    return math.exp(nll / T)


def test_dense_vs_prod_25bpc(tmp_path):
    """Dense vs TurboQuant (~2.5 bpc) perplexity comparison.

    Uses the tiny synthetic model by default (exercises the full pipeline).
    When both TQ_TEST_LLAMA_MODEL and TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY=1
    are set a real Llama model is loaded and the strict Δppl ≤ 1.5 gate is
    applied. The real-model path is treated as an exploratory batch-quality
    probe rather than a default integration contract.
    """
    from mlx_lm.models.cache import make_prompt_cache
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list

    _model_id = os.environ.get("TQ_TEST_LLAMA_MODEL", "")
    if _model_id:
        if os.environ.get(_REAL_MODEL_OPT_IN_ENV) != "1":
            pytest.skip(
                "real-model Δppl gate is exploratory; set "
                "TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY=1 to opt in"
            )
        from mlx_lm import load as _load
        model, tokenizer = _load(_model_id)
        max_delta_ppl = _MAX_DELTA_PPL_REAL
    else:
        from tests.helpers.tiny_model import TinyModel, TinyTokenizer
        model, tokenizer = TinyModel(), TinyTokenizer()
        mx.eval(model.parameters())
        max_delta_ppl = _MAX_DELTA_PPL_SYNTHETIC

    # ~2.5 bpc: k_bits=2, v_bits=3, both MSE (no residual)
    cfg = TurboQuantConfig(
        algorithm="turboquant_mse",
        k_bits=2, k_group_size=64,
        v_bits=3, v_group_size=64,
        rotation="hadamard",
        residual_mode="none",
    )

    delta_ppls = []
    for text in _SAMPLE_PROMPTS:
        tok_ids = list(getattr(tokenizer, "encode")(text))[: _MAX_TOKENS + 1]
        if len(tok_ids) < 4:
            continue
        input_ids = mx.array(tok_ids)[None]   # [1, T]
        targets = input_ids[0, 1:]
        feed = input_ids[:, :-1]

        # Dense forward
        dense_cache = make_prompt_cache(model)
        dense_logits = model(feed, cache=dense_cache)[0]
        mx.eval(dense_logits)
        dense_log_p = dense_logits - mx.logsumexp(
            dense_logits, axis=-1, keepdims=True
        )
        dense_ppl = _ppl(dense_log_p, targets)

        # TurboQuant 2.5 bpc forward
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
        f"at ~2.5 bpc (k_bits=2, v_bits=3)"
    )
