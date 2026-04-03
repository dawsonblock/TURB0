"""Integration test: dense vs TurboQuant (~3.5 bpc) perplexity gate.

Uses ``paper_mse`` preset (k_bits=3, k_group_size=64, v_bits=4, v_group_size=64)
which achieves approximately 3.75 bpc average.

Gate: mean Δppl ≤ 0.5 on a representative Llama-family model.

Skipped unless TQ_TEST_LLAMA_MODEL is set.
Set to a small model for fast iteration, e.g.:

    export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import math
import os

import pytest

_MODEL_ID = os.environ.get("TQ_TEST_LLAMA_MODEL", "")

mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

pytestmark = pytest.mark.skipif(
    not _MODEL_ID,
    reason=(
        "Dense vs prod 3.5bpc test disabled. "
        "Set TQ_TEST_LLAMA_MODEL to a small model, e.g. "
        "export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit"
    ),
)

_SAMPLE_PROMPTS = [
    "The theory of relativity was developed by Albert Einstein in the early twentieth century.",
    "In machine learning, transformers have revolutionised natural language processing tasks.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse",
]

_MAX_TOKENS = 64
_MAX_DELTA_PPL = 0.5


def _ppl(log_p, targets):
    import mlx.core as mx_
    T = targets.shape[0]
    nll = -float(mx_.sum(log_p[mx_.arange(T), targets]).item())
    return math.exp(nll / T)


def test_dense_vs_prod_35bpc(tmp_path):
    """Dense and TurboQuant (~3.5 bpc) perplexity must be within 0.5 PPL."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list

    model, tokenizer = load(_MODEL_ID)
    cfg = TurboQuantConfig.from_preset("paper_mse")

    delta_ppls = []
    for text in _SAMPLE_PROMPTS:
        tok_ids = tokenizer.encode(text)[:_MAX_TOKENS + 1]
        if len(tok_ids) < 4:
            continue
        input_ids = mx.array(tok_ids)[None]   # [1, T]
        targets = input_ids[0, 1:]
        feed = input_ids[:, :-1]
        T = int(targets.shape[0])

        # Dense
        dense_cache = make_prompt_cache(model)
        dense_logits = model(feed, cache=dense_cache)[0]
        mx.eval(dense_logits)
        dense_log_p = dense_logits - mx.logsumexp(dense_logits, axis=-1, keepdims=True)
        dense_ppl = _ppl(dense_log_p, targets)

        # TurboQuant
        tq_cache = make_prompt_cache(model)
        upgrade_cache_list(tq_cache, k_start=0, config=cfg)
        tq_logits = model(feed, cache=tq_cache)[0]
        mx.eval(tq_logits)
        tq_log_p = tq_logits - mx.logsumexp(tq_logits, axis=-1, keepdims=True)
        tq_ppl = _ppl(tq_log_p, targets)

        delta_ppls.append(tq_ppl - dense_ppl)

    assert delta_ppls, "No prompts produced valid results"
    mean_delta = sum(delta_ppls) / len(delta_ppls)
    assert mean_delta <= _MAX_DELTA_PPL, (
        f"Mean Δppl={mean_delta:.4f} exceeds gate {_MAX_DELTA_PPL} "
        f"at ~3.5 bpc (paper_mse preset)"
    )
