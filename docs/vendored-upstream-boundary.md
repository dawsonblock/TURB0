# Upstream mlx-lm Boundary

TurboQuant no longer ships a vendored `mlx_lm/` tree in this repository.

This is the canonical human architecture explainer. `VENDORED_MLX_LM.md`
remains only as a short continuity stub for compatibility with existing tests
and tooling.

Instead, TurboQuant integrates with upstream `mlx_lm` through a narrow
import-time/runtime patch layer. The canonical runtime entry point remains
`turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)`, and
`turboquant.patch.apply_mlx_lm_patches()` routes supported decode flows back to
that allowlisted upgrade path.

## Patched upstream hooks

- `mlx_lm.models.base.scaled_dot_product_attention`
- `mlx_lm.models.cache.make_prompt_cache`
- `mlx_lm.generate.generate_step`

## Supported scope

- Import/runtime patching of upstream `mlx_lm` hooks used by the bounded
  TurboQuant decode path.
- Support-gated cache promotion through `upgrade_cache_list(...)`.
- Allowlisted Apple-Silicon MLX runtime coverage for the `llama` and `gemma`
  families only.

## Unsupported

- Blanket support for every architecture reachable through upstream `mlx_lm`.
- Vendored-upstream maintenance inside this checkout.
- VLM support, multimodality, or non-Apple runtime claims.

## Current maintenance boundary

TurboQuant maintains its own runtime code under `turboquant/` plus the
compatibility shim surface that points callers back to the canonical
`upgrade_cache_list(...)` path. Upstream `mlx_lm` is patched in memory at
import/runtime; it is not copied into this repository.
