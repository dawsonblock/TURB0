# `mlx_lm` Integration Boundary

TurboQuant no longer ships a vendored `mlx_lm` tree.

## Current model

- TurboQuant depends on upstream `mlx_lm` at install/runtime on Apple Silicon.
- `turboquant.patch.apply_mlx_lm_patches()` applies an import-time monkey-patch
  layer to upstream `mlx_lm`.
- The canonical runtime entry point remains
  `turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)`.

## Patched upstream hooks

- `mlx_lm.models.base.scaled_dot_product_attention`
- `mlx_lm.models.cache.make_prompt_cache`
- `mlx_lm.generate.generate_step`

## Support boundary

The monkey-patch layer explains how TurboQuant integrates with upstream
`mlx_lm`, but it does not widen the TurboQuant support allowlist. The
allowlisted runtime path is still `upgrade_cache_list(...)`.
