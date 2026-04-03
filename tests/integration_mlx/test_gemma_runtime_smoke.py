"""
Gemma-family runtime smoke test.

Contract
--------
- Uses a tiny synthetic model (no download required).  If ``TQ_TEST_GEMMA_MODEL``
  is set the real model is used instead; that path is reserved for CI
  certification.
- Skipped on non-Apple-Silicon hosts (MLX not importable).
- Same contract as the Llama smoke test: prefill → TQ upgrade → decode → artifact.
- Certify the Llama smoke test before relying on Gemma results with a real model.

To run with a real Gemma model::

    export TQ_TEST_GEMMA_MODEL=mlx-community/gemma-2-2b-it-4bit
    python -m pytest tests/integration_mlx/test_gemma_runtime_smoke.py -v
"""
import json
import os
import time

import pytest

# Platform gate.
mx = pytest.importorskip("mlx.core", reason="Requires MLX (Apple Silicon)")

from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache  # noqa: E402


def test_gemma_runtime_smoke(tmp_path):
    """
    End-to-end Gemma-family smoke: prefill → TQ upgrade → decode → artifact.

    Uses a tiny synthetic model by default (no download needed).  Identical
    contract to test_llama_runtime_smoke.
    """
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.generate import generate_step

    _model_id = os.environ.get("TQ_TEST_GEMMA_MODEL", "")
    if _model_id:
        from mlx_lm import load as _load
        t_load = time.perf_counter()
        model, tokenizer = _load(_model_id)
        load_s = round(time.perf_counter() - t_load, 2)
    else:
        from tests.helpers.tiny_model import TinyModel, TinyTokenizer
        t_load = time.perf_counter()
        model, tokenizer = TinyModel(), TinyTokenizer()
        mx.eval(model.parameters())
        load_s = round(time.perf_counter() - t_load, 2)
        _model_id = "synthetic/TinyModel"

    prompt_text = "The TurboQuant cache compresses KV state by"
    prompt_ids = mx.array(tokenizer.encode(prompt_text))

    prompt_cache = make_prompt_cache(model)

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
        tokens_out.append(int(token))
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
        "model": _model_id,
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
