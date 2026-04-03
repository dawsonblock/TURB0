# Vendored `mlx_lm` — Fork Documentation

## Overview

TurboQuant ships a **vendored fork** of [mlx-lm](https://github.com/ml-explore/mlx-lm)
in the top-level `mlx_lm/` directory. This fork contains TurboQuant-specific patches
that wire the KV-cache compression runtime into model attention and generation loops.

## Fork Details

| Field | Value |
|---|---|
| **Upstream repo** | `ml-explore/mlx-lm` |
| **Upstream version at fork** | `0.29.1` (`mlx_lm/_version.py`) |
| **Fork date** | Pre-v0.2.0 TurboQuant release |
| **TurboQuant version** | `0.2.2` |
| **License** | MIT (Apple Inc. — preserved in source headers) |

## TurboQuant-Specific Patches

The following files diverge from upstream `mlx-lm`:

### `mlx_lm/models/cache.py`

- Added deprecated/internal `to_turboquant()` compatibility helper on `KVCache`
- `to_turboquant()` bypasses the TurboQuant model-family support gate; the canonical public path is `upgrade_cache_list(...)`
- Added `make_tq_config()` helper
- Added structured logging (`turboquant.cache`)

### `mlx_lm/models/base.py`

- `scaled_dot_product_attention(...)` type-guards on `TurboQuantKeysView`
- When a TQ view is detected, dispatches to `turboquant_streaming_attention()`
- This central dispatch path explains how upgraded caches are consumed, but it does **not** widen the supported-family allowlist

### `mlx_lm/models/llama.py`

- Llama-family attention remains wired for the allowlisted TQ path
- Added structured logging (`turboquant.llama`)

### `mlx_lm/models/gemma.py`

- Gemma-family reference wiring for the allowlisted TQ path
- Added structured logging (`turboquant.gemma`)

### `mlx_lm/models/gemma2.py`

- Gemma 2 attention path handles `TurboQuantKeysView` explicitly
- Preserves Gemma 2 logit soft-capping while dispatching to `turboquant_streaming_attention()`

### `mlx_lm/generate.py`

- Added `maybe_turboquant_k_cache()` (deprecated compatibility shim)
- `maybe_turboquant_k_cache()` delegates to `upgrade_cache_list(...)`
- After `upgrade_cache_list()`, logs one-time INFO confirming TQ cache activation
- Added structured logging (`turboquant.generate`)

### Files Unchanged from Upstream

All other files in `mlx_lm/` (e.g., `convert.py`, `lora.py`, `server.py`, `utils.py`,
model architectures other than `llama.py` and `gemma.py`) are **unmodified** from
upstream `v0.29.1`.

## How to Update from Upstream

> ⚠️ Manual merge required. There is no automated sync.

1. **Identify the target upstream version:**

   ```bash
   git clone https://github.com/ml-explore/mlx-lm.git /tmp/mlx-lm-upstream
   cd /tmp/mlx-lm-upstream
   git log --oneline -10
   ```

2. **Diff the patched files against the upstream equivalent:**

   ```bash
   diff -u /tmp/mlx-lm-upstream/mlx_lm/models/cache.py  mlx_lm/models/cache.py
   diff -u /tmp/mlx-lm-upstream/mlx_lm/models/base.py   mlx_lm/models/base.py
   diff -u /tmp/mlx-lm-upstream/mlx_lm/models/llama.py  mlx_lm/models/llama.py
   diff -u /tmp/mlx-lm-upstream/mlx_lm/models/gemma.py  mlx_lm/models/gemma.py
   diff -u /tmp/mlx-lm-upstream/mlx_lm/models/gemma2.py mlx_lm/models/gemma2.py
   diff -u /tmp/mlx-lm-upstream/mlx_lm/generate.py      mlx_lm/generate.py
   ```

3. **Apply upstream changes to unpatched files** (safe to overwrite):

   ```bash
   # Copy all files EXCEPT the patched ones
   rsync -av --exclude='models/cache.py' --exclude='models/base.py' \
             --exclude='models/llama.py' --exclude='models/gemma.py' \
             --exclude='models/gemma2.py' --exclude='generate.py' \
             
             /tmp/mlx-lm-upstream/mlx_lm/ mlx_lm/
   ```

4. **Manually merge patched files:** review upstream changes to `cache.py`,
   `base.py`, `llama.py`, `gemma.py`, `gemma2.py`, and `generate.py`, then apply them while preserving
   TQ dispatch blocks (search for `TurboQuant` or `turboquant` in each file).

5. **Update the version marker:**

   ```bash
   # In mlx_lm/_version.py — update to match the new upstream version
   ```

6. **Run the full test suite:**

   ```bash
   make test-mlx
   python -m pytest tests/integration_mlx/ -v --tb=short
   ```

7. **Update this document** with the new upstream version and date.

## Why Vendor?

TurboQuant requires modifications to the **inner attention loop** and **cache
data structures** of `mlx-lm`. These changes are not (yet) upstreamable because:

- The `TurboQuantKeysView` dispatch adds a new code path in vendored model/runtime files
- The cache upgrade APIs (`upgrade_cache_list(...)` and the deprecated compatibility helper `to_turboquant()`) modify `KVCache` internals
- Upstream `mlx-lm` does not have a plugin/hook system for cache backends

If upstream `mlx-lm` adds a cache-backend plugin API in the future, TurboQuant
should migrate to it and drop the vendored fork.

## Identifying Patched Sections

All TurboQuant-specific code blocks in the vendored files are marked with comments
or identifiable by:

- Imports from `turboquant.*`
- `isinstance(keys, TurboQuantKeysView)` checks
- Logger names starting with `turboquant.`
- Function calls to `turboquant_streaming_attention()`
