"""
Llama-family runtime smoke test.

Contract
--------
- Requires ``TQ_TEST_LLAMA_MODEL`` env var (small Llama HF model ID).
- Skipped on non-Apple platforms (MLX not importable) and when the env var
  is absent — these are not failures.
- Loads model + tokenizer via ``mlx_lm.load``.
- Creates the KV cache externally so it can be inspected post-generation.
- Calls ``generate_step`` with ``turboquant_k_start=0`` to trigger the cache
  upgrade on the very first prefill step.
- Asserts at least one cache layer is :class:`TurboQuantKCache` after
  generation — if none are upgraded the runtime silently took the dense path.
- Asserts at least one token was generated.
- Writes an artifact JSON with ``status="ok"`` and
  ``turboquant_active=True``.

Suggested model for fast iteration (< 1 GB download)::

    export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
    python -m pytest tests/integration_mlx/test_llama_runtime_smoke.py -v
"""
import json
import os
import time

import pytest

_MODEL_ID = os.environ.get("TQ_TEST_LLAMA_MODEL", "")

# Platform gate — entire module is skipped on non-Apple hosts.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

# Model gate — skip when the caller has not configured a model.
pytestmark = pytest.mark.skipif(
    not _MODEL_ID,
    reason=(
        "Llama smoke test disabled. "
        "Set TQ_TEST_LLAMA_MODEL to a small model, e.g. "
        "export TQ_TEST_LLAMA_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit"
    ),
)

from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache  # noqa: E402


def test_llama_runtime_smoke(tmp_path):
    """
    End-to-end Llama smoke: load → prefill → TQ upgrade → decode → artifact.

    Fails if generation routes through the dense (non-TurboQuant) path,
    i.e. no cache layers are TurboQuantKCache after generate_step.
    """
    from mlx_lm import load
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step

    # 1. Load model and tokenizer.
    t_load = time.perf_counter()
    model, tokenizer = load(_MODEL_ID)
    load_s = round(time.perf_counter() - t_load, 2)

    # 2. Encode a short prompt.
    prompt_text = "The TurboQuant cache compresses KV state by"
    prompt_ids = mx.array(tokenizer.encode(prompt_text))

    # 3. Create KV cache externally so we can inspect it after generation.
    prompt_cache = mlx_cache.make_prompt_cache(model)

    # 4. Run generate_step; turboquant_k_start=0 triggers the upgrade
    # immediately after the first prefill step.
    t_gen = time.perf_counter()
    tokens_out = []
    for token, _logprobs in generate_step(
        prompt_ids,
        model,
        max_tokens=5,
        prompt_cache=prompt_cache,
        turboquant_k_start=0,
        turboquant_k_bits=3,
        turboquant_group_size=32,
    ):
        tokens_out.append(int(token.item()))
    mx.eval()
    gen_s = round(time.perf_counter() - t_gen, 2)

    # 5. Verify at least one token was generated.
    assert len(tokens_out) > 0, "generate_step produced no tokens"

    # 6. Verify TurboQuant cache is active.
    # If no layer is TurboQuantKCache the runtime silently took the dense path.
    tq_layer_indices = [
        i for i, c in enumerate(prompt_cache) if isinstance(c, TurboQuantKCache)
    ]
    assert tq_layer_indices, (
        "Dense fallback detected: no cache layers were upgraded to "
        "TurboQuantKCache after generate_step with turboquant_k_start=0. "
        f"Cache types: {[type(c).__name__ for c in prompt_cache[:4]]}"
    )

    # 7. Write artifact for certification harness.
    artifact = {
        "status": "ok",
        "turboquant_active": True,
        "model": _MODEL_ID,
        "tokens_generated": len(tokens_out),
        "tq_layer_indices": tq_layer_indices,
        "total_layers": len(prompt_cache),
        "load_time_s": load_s,
        "gen_time_s": gen_s,
    }
    art_path = tmp_path / "smoke_llama.json"
    art_path.write_text(json.dumps(artifact, indent=2))

    assert artifact["status"] == "ok"
    assert artifact["turboquant_active"] is True
