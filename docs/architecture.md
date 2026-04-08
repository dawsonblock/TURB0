# TurboQuant Architecture

TurboQuant is a narrow Apple-Silicon MLX runtime path built around
`upgrade_cache_list(...)` and a monkey-patched upstream `mlx_lm` integration.

## 1. Canonical runtime path

The support-gated path is:

`upgrade_cache_list(...)` → `TurboQuantKCache.update_and_fetch(...)` →
`TurboQuantKeysView` → patched
`mlx_lm.models.base.scaled_dot_product_attention(...)` →
`turboquant_streaming_attention(...)`

Routing through `base.py` is not the same as being in the supported allowlist.
Only the hardcoded `SUPPORTED_FAMILIES` gate (`llama`, `gemma`) defines the
supported model-family boundary, and static tests keep it aligned with
`turboquant/contract.json`.

## 2. Primary components

### 2.1 Configuration

`turboquant/config.py` remains the runtime configuration source of truth.

- `v_enabled` is enabled by default (`True`).
- `block_tokens` remains a compatibility-only knob.
- The current streaming-attention hot path does not read `block_tokens` as a
  live runtime control.

### 2.2 Storage

`TurboQuantKVCache` remains the canonical storage engine.

- Encoded blocks are still preserved for serialization and state round-trips.
- The runtime attention path consumes append-only flat packed tensors and
  lightweight chunk metadata instead of iterating Python block objects.
- `TurboQuantKCache(...)` remains an internal / eval-only adapter around the
  canonical storage engine.

### 2.3 Attention fast path

`turboquant_streaming_attention(...)` now:

- rotates queries once per call for the scalar/QJL fast path
- scores flat K-history slices from runtime-packed tensors
- decodes V in chunks instead of concatenating the full dense V history
- uses online softmax with a running max, running denominator, and running
  accumulator (`log-sum-exp` style streaming reduction)
- keeps PolarQuant and top-k residual paths on the slower fallback path until
  their fused kernels are available

## 3. Upstream patch layer

TurboQuant no longer relies on a vendored `mlx_lm` tree.
`turboquant.patch.apply_mlx_lm_patches()` patches upstream `mlx_lm` in memory:

- `mlx_lm.models.base.scaled_dot_product_attention`
- `mlx_lm.models.cache.make_prompt_cache`
- `mlx_lm.generate.generate_step`

That patch layer explains integration behavior, but it does not widen the
support contract.

## 4. Support boundary

- allowlisted families are hardcoded in `turboquant/runtime/support.py`
  and statically checked against `turboquant/contract.json`
- direct adapter construction remains secondary
- experimental Metal kernels remain optional acceleration paths
- public static CI checks buildability and static coherence, not Apple runtime
  proof
