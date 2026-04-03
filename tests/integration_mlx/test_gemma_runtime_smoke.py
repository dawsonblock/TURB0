"""
Gemma-family runtime smoke test.

Contract
--------
- Requires ``TQ_TEST_GEMMA_MODEL`` env var (small Gemma HF model ID).
- Skipped on non-Apple platforms (MLX not importable) and when the env var
  is absent — these are not failures.
- Same contract as the Llama smoke test: load → prefill → TQ upgrade →
  decode → artifact.
- Do not run this test until the Llama smoke test (``test_llama_runtime_smoke``)
  has been validated and artifact-backed. Gemma is the second model family;
  certify Llama first.

Suggested model for fast iteration::

    export TQ_TEST_GEMMA_MODEL=mlx-community/gemma-2-2b-it-4bit
    python -m pytest tests/integration_mlx/test_gemma_runtime_smoke.py -v
"""
import json
import os
import time

import pytest

_MODEL_ID = os.environ.get("TQ_TEST_GEMMA_MODEL", "")

# Platform gate.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

# Model gate.
pytestmark = pytest.mark.skipif(
    not _MODEL_ID,
    reason=(
        "Gemma smoke test disabled. "
        "Set TQ_TEST_GEMMA_MODEL to a small model, e.g. "
        "export TQ_TEST_GEMMA_MODEL=mlx-community/gemma-2-2b-it-4bit"
    ),
)

from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache  # noqa: E402


def test_gemma_runtime_smoke(tmp_path):
    """
    End-to-end Gemma smoke: load → prefill → TQ upgrade → decode → artifact.

    Identical contract to test_llama_runtime_smoke. Run Llama smoke first;
    validate and produce Llama artifacts before attempting Gemma certification.
    """
    from mlx_lm import load
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step

    t_load = time.perf_counter()
    model, tokenizer = load(_MODEL_ID)
    load_s = round(time.perf_counter() - t_load, 2)

    prompt_text = "The TurboQuant cache compresses KV state by"
    prompt_ids = mx.array(tokenizer.encode(prompt_text))

    prompt_cache = mlx_cache.make_prompt_cache(model)

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

    assert len(tokens_out) > 0, "generate_step produced no tokens"

    tq_layer_indices = [
        i for i, c in enumerate(prompt_cache) if isinstance(c, TurboQuantKCache)
    ]
    assert tq_layer_indices, (
        "Dense fallback detected: no cache layers upgraded to TurboQuantKCache. "
        f"Cache types: {[type(c).__name__ for c in prompt_cache[:4]]}"
    )

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
    art_path = tmp_path / "smoke_gemma.json"
    art_path.write_text(json.dumps(artifact, indent=2))

    assert artifact["status"] == "ok"
    assert artifact["turboquant_active"] is True
