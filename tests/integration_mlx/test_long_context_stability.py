"""
Long-context stability smoke test.

Verifies that the TurboQuant KV cache remains stable when the context
exceeds a single compression block (default ``block_tokens=256``). This is
NOT a quality-correctness test — it does not check output text, perplexity,
or KL divergence. It only checks that:

- Generation completes without raising an exception.
- At least one token is produced from a >300-token prompt.
- No NaN logprobs are emitted (overflow / underflow guard).
- At least one cache layer remains ``TurboQuantKCache`` after generation.

Uses a tiny synthetic model by default (no download required).  If
``TQ_TEST_LLAMA_MODEL`` is set the real Llama model is used instead.

To run with a real model::

    export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
"""
import os

import pytest

# Platform gate.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache  # noqa: E402

_PROMPT_TOKENS = 320  # target prompt length — exceeds default block_tokens=256


def test_long_context_stability(tmp_path):
    """
    Generate tokens from a >300-token prompt with TurboQuantKCache active.

    Uses a tiny synthetic model by default; a real model is used when
    TQ_TEST_LLAMA_MODEL is set.  Passes if:
      - Generation completes without exception.
      - At least one token is generated.
      - No NaN logprobs are produced.
      - At least one cache layer is TurboQuantKCache at the end.
    """
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    _model_id = os.environ.get("TQ_TEST_LLAMA_MODEL", "")
    if _model_id:
        from mlx_lm import load as _load
        model, tokenizer = _load(_model_id)
        # Build a >300-token prompt the same way the original test did.
        seed = "The quick brown fox jumps over the lazy dog. "
        long_text = seed * ((_PROMPT_TOKENS // len(seed.split())) + 2)
        prompt_ids_list = tokenizer.encode(long_text)[:_PROMPT_TOKENS]
    else:
        from tests.helpers.tiny_model import TinyModel, TinyTokenizer
        model, tokenizer = TinyModel(), TinyTokenizer()
        mx.eval(model.parameters())
        # With TinyTokenizer one char = one token; repeat a phrase to get 320 tokens.
        seed = "The quick brown fox jumps over the lazy dog. "
        long_text = seed * 8  # 8 × 45 chars = 360 chars → 360 tokens
        prompt_ids_list = tokenizer.encode(long_text)[:_PROMPT_TOKENS]

    assert len(prompt_ids_list) >= 300, (
        f"Could not build a >{_PROMPT_TOKENS}-token prompt; "
        f"got {len(prompt_ids_list)} tokens."
    )
    prompt_ids = mx.array(prompt_ids_list)

    prompt_cache = make_prompt_cache(model)

    tokens_out = []
    for token, logprobs in generate_step(
        prompt_ids,
        model,
        max_tokens=8,
        prompt_cache=prompt_cache,
        turboquant_k_start=0,
        turboquant_k_bits=3,
        turboquant_group_size=32,
        turboquant_model_family="llama",
    ):
        tok = int(token)
        assert tok >= 0, f"Negative/invalid token id produced: {tok}"

        lp_max = float(mx.max(logprobs).item())
        assert lp_max == lp_max, (  # NaN != NaN
            "NaN log-probability detected during long-context generation — "
            "possible overflow in the TurboQuant attention or dequant path."
        )
        tokens_out.append(tok)

    mx.eval()

    assert len(tokens_out) > 0, (
        f"No tokens generated from a {len(prompt_ids_list)}-token prompt."
    )

    tq_layers = [c for c in prompt_cache if isinstance(c, TurboQuantKCache)]
    assert tq_layers, (
        "Dense fallback detected after long-context generation: "
        "no TurboQuantKCache layers remain in prompt_cache."
    )
