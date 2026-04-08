# `mlx_lm` Integration Boundary

This filename is retained for continuity and tooling compatibility, but this
checkout does not vendor `mlx_lm`.

This document is the short continuity boundary stub. The canonical human
architecture explainer is
[`docs/vendored-upstream-boundary.md`](docs/vendored-upstream-boundary.md).

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

## Active repo touchpoints

- `turboquant/patch.py`
- `turboquant/integrations/mlx/upgrade.py`

## Support boundary

The monkey-patch layer explains how TurboQuant integrates with upstream
`mlx_lm`, but it does not widen the TurboQuant support allowlist. The
allowlisted runtime path is still `upgrade_cache_list(...)`.
