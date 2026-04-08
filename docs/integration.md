# TurboQuant Integration Guide

## 0. Canonical supported runtime path

For supported runtime use, the canonical entry point is
`upgrade_cache_list(..., model_family=...)`.

TurboQuant patches upstream `mlx_lm` at import time so that upgraded caches are
consumed through patched `scaled_dot_product_attention(...)`,
`make_prompt_cache(...)`, and `generate_step(...)`.

This dispatch detail explains how an already-upgraded cache is consumed. It is
not itself the support contract. The cache upgrade gate (`upgrade_cache_list`)
separately enforces the model-family allowlist: only `"llama"` and `"gemma"`
are in the supported set. Routing through `base.py` is not the same as being in the supported allowlist.

## 1. Concepts

TurboQuant inserts itself into two places:

1. **KV cache** — promote dense cache entries through `upgrade_cache_list(...)`
2. **Attention** — dispatch to the streaming attention path when the key tensor
   is a `TurboQuantKeysView`

## 2. Cache upgrade

Use `upgrade_cache_list(...)` directly.

Direct `TurboQuantKCache(...)` construction and
`_collect_logits_compressed()` remain internal/eval-only escape hatches, not
peer public runtime surfaces.
