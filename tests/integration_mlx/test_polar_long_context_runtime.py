"""PolarQuant long-context integration smoke test.

Contract
--------
- Uses a tiny synthetic model by default; if ``TQ_TEST_LLAMA_MODEL`` is set,
  a real Llama-family model is used instead.
- Builds a >300-token prompt, runs dense prefill, upgrades the cache via
  ``upgrade_cache_list(...)`` with ``TurboQuantConfig.polarquant_exp(...)``,
  and performs a few greedy decode steps against the upgraded cache.
- Asserts that at least one cache layer becomes ``TurboQuantKCache``, that the
  upgraded blocks carry PolarQuant payloads, and that decode logits remain
  finite after the upgrade.
- This covers the supported non-paper-facing PolarQuant runtime path. The
    Apple certification harness runs it as the Llama-scoped Polar smoke stage.
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

from turboquant.config import TurboQuantConfig  # noqa: E402
from turboquant.integrations.mlx.cache_adapter import (  # noqa: E402
    TurboQuantKCache,
)
from turboquant.integrations.mlx.upgrade import (  # noqa: E402
    upgrade_cache_list,
)

_PROMPT_TOKENS = 320
_DECODE_STEPS = 4
_TINY_MODEL_ID = "synthetic/TinyModel"


def _load_model_and_tokenizer():
    model_id = os.environ.get("TQ_TEST_LLAMA_MODEL", "")
    if model_id:
        from mlx_lm import load as _load

        model, tokenizer = _load(model_id)
        return model, tokenizer, model_id

    from tests.helpers.tiny_model import TinyModel, TinyTokenizer

    model = TinyModel()
    tokenizer = TinyTokenizer()
    mx.eval(model.parameters())
    return model, tokenizer, _TINY_MODEL_ID


def _build_prompt_ids(tokenizer, *, use_real_model: bool):
    seed = "The quick brown fox jumps over the lazy dog. "
    if use_real_model:
        long_text = seed * ((_PROMPT_TOKENS // len(seed.split())) + 2)
    else:
        long_text = seed * 8  # 8 x 45 chars = 360 char-level tokens

    prompt_ids = tokenizer.encode(long_text)[:_PROMPT_TOKENS]
    assert len(prompt_ids) >= 300, (
        f"Could not build a >{_PROMPT_TOKENS}-token prompt; "
        f"got {len(prompt_ids)} tokens."
    )
    return mx.array(prompt_ids)


def test_polar_long_context_runtime_path():
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer, model_id = _load_model_and_tokenizer()
    prompt_ids = _build_prompt_ids(
        tokenizer,
        use_real_model=model_id != _TINY_MODEL_ID,
    )
    prompt_cache = make_prompt_cache(model)

    logits = model(prompt_ids[None], cache=prompt_cache)
    mx.eval(logits)

    cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
    events = upgrade_cache_list(
        prompt_cache,
        k_start=0,
        config=cfg,
        model_family="llama",
    )

    assert any(event.upgraded for event in events), (
        "Dense prefill was not promoted to the PolarQuant runtime path."
    )

    tq_layers = [
        cache for cache in prompt_cache if isinstance(cache, TurboQuantKCache)
    ]
    assert tq_layers, "No cache layers were upgraded to TurboQuantKCache."
    assert any(
        layer._impl.block(0).polar is not None for layer in tq_layers
    ), (
        "Upgraded cache layers did not persist any PolarQuant payloads."
    )
    assert tq_layers[0]._impl.state()["algorithm"] == "polarquant_exp"

    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    tokens_out: list[int] = []
    for _ in range(_DECODE_STEPS):
        logits = model(next_token[:, None], cache=prompt_cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(logits, next_token)

        assert not bool(mx.any(mx.isnan(logits)).item()), (
            "PolarQuant decode produced NaN logits during streaming decode."
        )
        assert not bool(mx.any(mx.isinf(logits)).item()), (
            "PolarQuant decode produced Inf logits during streaming decode."
        )
        tokens_out.append(int(next_token.item()))

    assert len(tokens_out) == _DECODE_STEPS
    assert sum(1 for _ in tq_layers[0]._impl.iter_blocks()) >= 2, (
        "Streaming decode did not append any additional PolarQuant blocks."
    )
