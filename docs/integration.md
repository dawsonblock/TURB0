# TurboQuant Integration Guide

> How to wire TurboQuant into an mlx-lm model family.

---

## 0. Preferred approach — centralized dispatch (no per-model changes)

The simplest integration path requires **no changes to individual model files**.
`mlx_lm/models/base.py`’s `scaled_dot_product_attention` function now contains
a type-guard that automatically routes any `TurboQuantKeysView` tensor to
`turboquant_streaming_attention`:

```python
# mlx_lm/models/base.py (already applied)
def scaled_dot_product_attention(queries, keys, values, cache, scale, mask=None, ...):
    if type(keys).__name__ == "TurboQuantKeysView":
        from turboquant.runtime.attention import turboquant_streaming_attention
        return turboquant_streaming_attention(queries, keys, scale=scale, mask=mask)
    ...
```

This means the **attention dispatch** routes automatically for any model that calls
`scaled_dot_product_attention` from `base.py` — but this is **not the same as being supported**.
The cache upgrade gate (`upgrade_cache_list`) separately enforces the model-family allowlist:
only `"llama"` and `"gemma"` are in the certified allowlist. Routing through `base.py` does not
bring a model into the supported set. Verified on Llama and Gemma on Apple Silicon
(commit `6afc966`). All other architectures are unsupported or exploratory regardless of
whether they call `scaled_dot_product_attention`.

---

## 1. Concepts

TurboQuant inserts itself into two places:

1. **KV cache** — replace `KVCache` with `TurboQuantKCache` after prefill
2. **Attention** — dispatch to the streaming attention path when the key tensor
   is a `TurboQuantKeysView`

Both hooks are model-agnostic.  The cache upgrade is done once (after prefill);
the attention dispatch is a one-liner inside each model's attention `__call__`.

---

## 2. Cache upgrade

### Recommended (programmatic)

```python
from mlx_lm.models.cache import make_prompt_cache
from turboquant.integrations.mlx.upgrade import upgrade_cache_list
from turboquant.config import TurboQuantConfig

cache = make_prompt_cache(model)
# ... run prefill ...
cfg = TurboQuantConfig(k_bits=3, k_group_size=64, rotation="hadamard")
events = upgrade_cache_list(
    cache, k_start=64, config=cfg,
    model_family="llama",  # required — must be "llama" or "gemma" (the certified allowlist)
)
# decode loop continues with TurboQuant cache
```text
`upgrade_cache_list` returns a list of `CacheUpgradeEvent` objects (one per
layer) with `upgraded`, `layer_index`, `old_type`, `new_type`, `offset_at_upgrade`.

### Legacy (deprecated)

`mlx_lm.generate.maybe_turboquant_k_cache` is still importable for backward
compatibility but internally delegates to `upgrade_cache_list`.  New code should
use `upgrade_cache_list` directly.

---

## 3. Attention dispatch

### Step 1 — imports

Add these imports to your model file:

```python
from turboquant.runtime.attention import turboquant_streaming_attention
from turboquant.runtime.kv_interface import TurboQuantKeysView
```text
### Step 2 — dispatch inside attention `__call__`

Replace the attention call with:

```python
def __call__(self, x, mask=None, cache=None):
    q, k, v = ...   # project x

    if cache is not None:
        k, v = cache.update_and_fetch(k, v)

    scale = self.scale   # or head_dim ** -0.5
    
    if isinstance(k, TurboQuantKeysView):
        attn_out = turboquant_streaming_attention(
            q, k, v, mask=mask, scale=scale
        )
    else:
        # your existing dense attention implementation
        attn_out = ...

    return self.o_proj(attn_out)
```text
### Gemma example

`mlx_lm/models/gemma.py` is the reference implementation.  Search for
`turboquant_streaming_attention` to see the exact wiring.

---

## 4. Config mapping (legacy → production)

If you have old code using `TurboQuantConfig(main_bits=3, group_size=64, ...)`:

| legacy field | production field | notes |
|---|---|---|
| `main_bits` | `k_bits` | |
| `group_size` | `k_group_size` | |
| `rotation` | `rotation` | same values |
| — | `quantizer_mode` | `"scalar"` (default) or `"polar"` — see §4a below |
| `return_mode` | — | adapter-only; production upgrade path always uses streaming view mode |
| `resid_scale_bits` | — | adapter metadata only; production residual behavior is `residual_topk` |
| `residual` | — | ignored |
| `v_bits` | `v_bits` | |
| `v_group_size` | `v_group_size` | |
| `block_tokens` | `block_tokens` | accepted but not currently used in the attention dispatch path |

`mlx_lm.models.cache.TurboQuantConfig` is a legacy shim that performs this
mapping automatically.

---

## 4a. PolarQuant mode

Set `quantizer_mode="polar"` to use the `PolarQuantizer` (arXiv:2502.02617) instead of
`GroupScalarQuantizer` as the main K-cache quantiser.  All other configuration
fields (`rotation`, `residual_mode`, etc.) behave identically.

```python
from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline

# Drop-in replacement for scalar mode
cfg = TurboQuantConfig(
    quantizer_mode="polar",   # <-- switch here
    rotation="hadamard",       # recommended — makes angle distribution uniform
    residual_mode="none",      # polar encodes residual implicitly; QJL optional
)
pipe = TurboQuantPipeline(cfg)

block = pipe.encode_k(k_rotated)   # returns EncodedKeyBlock with .polar set
k_hat = pipe.decode_k(block)       # reconstructs via polar inverse
```

Or use the lower-level API directly:

```python
from turboquant.core.polar_quant import PolarQuantizer

pq = PolarQuantizer(n_levels=4, bits_l1=4, bits_le=2)
payload = pq.encode(x)   # → PolarQuantPayload (angle codes + final radii)
x_hat  = pq.decode(payload)
```

**When to prefer PolarQuant:**
- Encode-latency sensitive workloads (7–18× faster encode than scalar).
- Lower reconstruction error is required at the same bit-budget (~40% lower MSE at 3.875 bits/dim vs 3-bit scalar).
- Memory overhead from scale factors is undesirable (PolarQuant stores zero scales).

**Limitations:**
- Decode is slightly slower than scalar at large sequence lengths (interleaved reconstruction).
- Does not currently support QJL residual correction (residual is skipped).
- `paper_faithful_mode=True` is a deprecated stub; use `quantizer_mode="polar"` directly.

---

## 5. Llama wiring

`mlx_lm/models/llama.py` wires `turboquant_streaming_attention` identically to
Gemma.  See `mlx_lm/models/llama.py` → `Attention.__call__`.

---

## 6. Adding a new model family

**Attention dispatch routing (no wiring required for base.py models):** If the model's
attention calls `scaled_dot_product_attention` from `mlx_lm.models.base`, the attention
dispatch routes to `turboquant_streaming_attention` automatically when a `TurboQuantKeysView`
key is present. However, this does **not** mean the model is supported — the cache upgrade
gate (`upgrade_cache_list`) will raise `UnsupportedModelError` unless the family is added to
`SUPPORTED_FAMILIES` in `turboquant/runtime/support.py`. **Routing through `base.py` ≠
membership in the supported allowlist.** Currently only `"llama"` and `"gemma"` are in the
allowlist.

**Manual wiring (fallback):** For models with custom attention implementations
that bypass `base.py`:

1. Add `from turboquant.runtime.attention import turboquant_streaming_attention` and `from turboquant.runtime.kv_interface import TurboQuantKeysView`
2. In `Attention.__call__`, replace the dense `scaled_dot_product_attention`
   call with a manual check for `isinstance(k, TurboQuantKeysView)` and call `turboquant_streaming_attention`.
3. No changes to the cache object are needed — `TurboQuantKCache` is fully
   encapsulated.

---

## 7. Common integration pitfalls

| Pitfall | Fix |
|---|---|
| Positional `TurboQuantKVCache(cfg)` fails | `config` is positional-or-keyword; both `TurboQuantKVCache(cfg)` and `TurboQuantKVCache(config=cfg)` are valid |
| `cache._impl.config` hardcoded in attention | Use `impl = getattr(cache, "_impl", cache)` to support both `TurboQuantKCache` and `TurboQuantKVCache` |
| `quantize_main` / `dequantize_main` required | Since v0.2.2 these are optional; a `GroupScalarQuantizer` is auto-created from `config.k_bits` / `config.k_group_size` |
| Using PolarQuant with residual | PolarQuant sets `residual_mode` to `"none"` internally; set `residual_mode="none"` explicitly in config to avoid mismatch noise |
| `paper_faithful_mode=True` has no effect | It is a deprecated dead stub. Use `quantizer_mode="polar"` instead |

---

## 8. Testing

```bash
# Static tests (no MLX required)
pytest tests/unit_static/

# MLX unit tests (Apple Silicon only)
pytest tests/unit_mlx/

# Integration tests (Apple Silicon only)
pytest tests/integration_mlx/
```
