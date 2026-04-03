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

NOTE: This test reuses the Llama model from ``TQ_TEST_LLAMA_MODEL``. It
will be skipped if that variable is not set. The quality gate belongs to
``run_quality_eval.py`` (currently unimplemented).

Suggested model::

    export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
"""
import os

import pytest

_MODEL_ID = os.environ.get("TQ_TEST_LLAMA_MODEL", "")

# Platform gate.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

# Model gate.
pytestmark = pytest.mark.skipif(
    not _MODEL_ID,
    reason=(
        "Long-context stability test requires TQ_TEST_LLAMA_MODEL. "
        "Set to a small model, e.g. "
        "export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit"
    ),
)

from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache  # noqa: E402

_PROMPT_TOKENS = 320  # target prompt length — exceeds default block_tokens=256


def test_long_context_stability(tmp_path):
    """
    Generate tokens from a >300-token prompt with TurboQuantKCache active.

    Does not validate output quality (that is run_quality_eval.py's job).
    Passes if:
      - Generation completes without exception.
      - At least one token is generated.
      - No NaN logprobs are produced.
      - At least one cache layer is TurboQuantKCache at the end.
    """
    from mlx_lm import load
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step

    model, tokenizer = load(_MODEL_ID)

    # Build a prompt that is longer than one TQ block (block_tokens=256).
    seed = "The quick brown fox jumps over the lazy dog. "
    long_text = (seed * ((_PROMPT_TOKENS // len(seed.split())) + 2))
    prompt_ids_list = tokenizer.encode(long_text)[:_PROMPT_TOKENS]
    assert len(prompt_ids_list) >= 300, (
        f"Could not build a >{_PROMPT_TOKENS}-token prompt; "
        f"got {len(prompt_ids_list)} tokens. Adjust the seed text."
    )
    prompt_ids = mx.array(prompt_ids_list)

    prompt_cache = mlx_cache.make_prompt_cache(model)

    tokens_out = []
    for token, logprobs in generate_step(
        prompt_ids,
        model,
        max_tokens=8,
        prompt_cache=prompt_cache,
        turboquant_k_start=0,
        turboquant_k_bits=3,
        turboquant_group_size=32,
    ):
        tok = int(token.item())
        # NaN token id is a signal of arithmetic overflow in the TQ path.
        assert tok >= 0, f"Negative/invalid token id produced: {tok}"

        # Check for NaN in log-probabilities.
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
