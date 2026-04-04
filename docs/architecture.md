# TurboQuant Architecture

> **Status**: prototype, not production-ready  
> **Last updated**: April 2026

TurboQuant is an Apple-Silicon MLX prototype for compressing KV caches during
decode. This document describes the runtime path that actually exists in the
repo today. It is not a history of every design idea and it is not a support
claim beyond the narrow allowlisted path.

## 1. Canonical runtime path

The only supported runtime promotion path is:

`mlx_lm.generate.generate_step` → `_infer_model_family(...)` →
`maybe_turboquant_k_cache(...)` → `upgrade_cache_list(...)` →
`TurboQuantKCache.update_and_fetch(...)` → `TurboQuantKeysView` →
`mlx_lm.models.base.scaled_dot_product_attention(...)` →
`turboquant_streaming_attention(...)`

What that means in practice:

1. `generate_step` infers a family name or returns `None`.
2. `maybe_turboquant_k_cache(...)` decides whether to call `upgrade_cache_list(...)`.
3. `upgrade_cache_list(...)` enforces the allowlist in `turboquant/runtime/support.py`
   before mutating any cache entry.
4. Upgraded layers use `TurboQuantKCache.update_and_fetch(...)` to return a
   `TurboQuantKeysView` into the attention path.
5. `scaled_dot_product_attention(...)` type-guards on `TurboQuantKeysView` and
   routes decode to `turboquant_streaming_attention(...)`.

If this document disagrees with the allowlist or the upgrade gate, the code
wins. Vendored model files outside the allowlist do not become supported just
because they exist in `mlx_lm/`.

## 2. Primary components

### 2.1 TurboQuantConfig

`turboquant/config.py` is the single source of truth for runtime configuration.

- `v_enabled` is enabled by default (`True`).
- `block_tokens` is a compatibility-only knob retained for older configs and
  historical benchmarks.
- The current streaming-attention hot path does not read `block_tokens` as a
  live runtime control.

### 2.2 Cache upgrade and storage surfaces

| Surface | Status | Notes |
|---|---|---|
| `upgrade_cache_list(...)` | Canonical runtime path | Enforces the model-family allowlist before promotion |
| `TurboQuantKVCache` | Canonical storage engine | Owns the compressed runtime representation |
| `TurboQuantKCache(...)` | Internal / eval-only adapter | Thin `mlx_lm` protocol adapter around the canonical storage engine |
| `KVCache._to_turboquant()` | Private eval-only implementation | Primary bypass implementation; constructs `TurboQuantKCache` directly without the support gate |
| `KVCache.to_turboquant()` | Deprecated public alias | Delegates to `_to_turboquant()`; retained only for compatibility and interop |
| `_collect_logits_compressed()` | Private eval-only bypass | Direct adapter construction for dense-vs-compressed comparison |

### 2.3 Attention dispatch

`TurboQuantKCache.update_and_fetch(...)` appends compressed state and returns a
`TurboQuantKeysView`. `mlx_lm.models.base.scaled_dot_product_attention(...)`
type-guards on that view and routes to `turboquant_streaming_attention(...)`.

Inside `turboquant_streaming_attention(...)`:

- queries are rotated per block via `FixedRotation.apply()` inside `score_block()`
- compressed key/value blocks are decoded block by block
- score tensors are concatenated across blocks
- one `mx.softmax` is applied over the full key dimension

That is the runtime behavior that exists today.

## 3. Decode flow

```text
dense KVCache list
    │
    ├─ maybe_turboquant_k_cache(...)
    │      └─ upgrade_cache_list(...)
    │             └─ TurboQuantKCache adapters installed for allowlisted families
    │
decode step
    └─ TurboQuantKCache.update_and_fetch(...)
           └─ TurboQuantKeysView
                  └─ scaled_dot_product_attention(...)
                         └─ turboquant_streaming_attention(...)
```

Direct construction helpers exist, but the runtime contract should be read from
the path above, not from compatibility or eval entry points.

## 4. Event architecture

TurboQuant intentionally keeps runtime upgrade results separate from persistence
artifacts.

- `turboquant.integrations.mlx.upgrade.CacheUpgradeEvent` is the lightweight
  runtime result type returned by `upgrade_cache_list(...)`.
- `turboquant.runtime.events.CacheUpgradeEvent`, `UpgradeFailureEvent`, and
  `EventLog` belong to the optional persistence layer.
- `mlx_lm.generate.maybe_turboquant_k_cache(...)` currently discards the
  returned runtime events.
- The canonical decode path does not automatically persist `events.jsonl`.
- `turboquant.runtime.events.record_runtime_upgrade_events(...)` is the thin
  explicit adapter for certification or benchmark flows that want to persist
  runtime upgrade decisions.
- `turboquant.metrics.tracker.MetricsTracker.write(event_log=...)` is the
  manual boundary for JSONL artifact generation.

This split is intentional. Runtime upgrade decisions and certification logging
are related, but they are not the same surface.

## 5. Support boundary

The support claim remains narrow:

- only allowlisted families in `SUPPORTED_FAMILIES` are supported today: `llama`
  and `gemma`
- vendored model files outside that allowlist are unsupported
- direct adapter construction and compatibility helpers remain secondary
  surfaces, not peer public APIs
- custom Metal kernels and related acceleration experiments are optional and
  experimental
- public static CI does not certify MLX runtime behavior

## 6. Limitations

- Hadamard rotation is dense `O(d²)`, not butterfly `O(d log d)`.
- `TurboQuantKCache(...)`, `KVCache._to_turboquant()`, `KVCache.to_turboquant()`, and
  `_collect_logits_compressed()` bypass the canonical upgrade gate and should
  not be treated as peer runtime APIs.
- For custom attention implementations that do not call
  `scaled_dot_product_attention(...)`, the legacy `maybe_turboquant_attention(...)`
  helper remains a compatibility fallback, not a primary integration path.
- `block_tokens` remains public only for compatibility; it is not a live tuning
  lever in the current attention dispatch path.
- Real runtime validation still requires Apple Silicon hardware plus
  retained local evidence directories or published workflow artifacts.

