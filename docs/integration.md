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

This means **any** model that calls `scaled_dot_product_attention` from `base.py`
will automatically support TurboQuant without further modification. Verified on
Llama and Gemma on Apple Silicon (commit `6afc966`). Qwen is currently unsupported and exploratory.

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
events = upgrade_cache_list(cache, k_start=64, config=cfg)
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
| `return_mode` | — | adapter-only; production upgrade path always uses streaming view mode |
| `resid_scale_bits` | — | adapter metadata only; production residual behavior is `residual_topk` |
| `residual` | — | ignored |
| `v_bits` | `v_bits` | |
| `v_group_size` | `v_group_size` | |
| `block_tokens` | `block_tokens` | |

`mlx_lm.models.cache.TurboQuantConfig` is a legacy shim that performs this
mapping automatically.

---

## 5. Llama wiring

`mlx_lm/models/llama.py` wires `turboquant_streaming_attention` identically to
Gemma.  See `mlx_lm/models/llama.py` → `Attention.__call__`.

---

## 6. Adding a new model family

**Preferred (zero changes required):** If the model’s attention calls
`scaled_dot_product_attention` from `mlx_lm.models.base`, TurboQuant already
works out of the box via the centralized dispatch in `base.py`.

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
| `GroupScalarQuantizer(bits=...)` | Use `n_bits=` — the constructor parameter is `n_bits`, not `bits` |
| Positional `TurboQuantKVCache(cfg)` fails | `config` is positional-or-keyword; both `TurboQuantKVCache(cfg)` and `TurboQuantKVCache(config=cfg)` are valid |
| `cache._impl.config` hardcoded in attention | Use `impl = getattr(cache, "_impl", cache)` to support both `TurboQuantKCache` and `TurboQuantKVCache` |
| `quantize_main` / `dequantize_main` required | Since v0.2.2 these are optional; a `GroupScalarQuantizer` is auto-created from `config.k_bits` / `config.k_group_size` |

---

## 8. Testing

```bash
# Unit tests (turboquant package)
pytest tests/unit/

# Integration tests (mlx_lm + turboquant)
pytest tests/integration/

# Full suite
pytest tests/
```
