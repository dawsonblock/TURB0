<!-- Generated from turboquant/contract.json by scripts/render_support_contract.py. Do not edit by hand. -->
# Preset Modes

This preset reference maps the runtime preset surface to the contract labels used by the repo.
It does not widen the supported product contract. Paper-facing, supported non-paper-facing, and compatibility-only presets stay explicitly separated here.

## Stable Preset Table

| Preset | Classification | Canonical preset | Algorithm family | Quantizer | Residual | Stable constructor | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `paper_mse` | paper-facing | `paper_mse` | `paper_mse` | `scalar` | `none` | `TurboQuantConfig.from_preset("paper_mse")` | Paper-facing MSE stage. |
| `paper_prod_qjl` | paper-facing | `paper_prod_qjl` | `paper_prod_qjl` | `scalar` | `qjl` | `TurboQuantConfig.from_preset("paper_prod_qjl")` | Primary paper-facing two-stage preset using a 1-bit QJL residual. |
| `paper_prod` | paper-facing | `paper_prod_qjl` | `paper_prod_qjl` | `scalar` | `qjl` | `TurboQuantConfig.from_preset("paper_prod")` | Paper-facing production-style preset using a 1-bit QJL residual. |
| `polarquant_exp` | supported non-paper-facing | `polarquant_exp` | `polarquant_exp` | `polar` | `none` | `TurboQuantConfig.from_preset("polarquant_exp")` | Supported non-paper-facing PolarQuant branch with family-scoped runtime and quality certification. |
| `legacy_topk` | compatibility-only | `legacy_topk` | `legacy_topk` | `scalar` | `topk` | `TurboQuantConfig.from_preset("legacy_topk")` | Explicit legacy top-k compatibility preset; not part of the paper-facing contract. |
| `high_compression` | compatibility-only | `paper_prod_qjl` | `paper_prod_qjl` | `scalar` | `qjl` | `TurboQuantConfig.from_preset("high_compression")` | Legacy convenience alias for the QJL production-style preset. |
| `balanced` | compatibility-only | `balanced` | `legacy_topk` | `scalar` | `topk` | `TurboQuantConfig.from_preset("balanced")` | Legacy top-k compatibility preset; not part of the paper-facing contract. |
| `max_quality` | compatibility-only | `max_quality` | `legacy_topk` | `scalar` | `topk` | `TurboQuantConfig.from_preset("max_quality")` | Legacy top-k compatibility preset; not part of the paper-facing contract. |

## Recommended Comparison Set

- `paper_mse` — scalar-only paper baseline.
- `paper_prod_qjl` — primary two-stage paper-facing preset.
- `polarquant_exp` — closest available first-stage-only non-paper research path.
- `legacy_topk`, `balanced`, `max_quality`, and `high_compression` — compatibility-only surfaces for historical comparisons and older config loading.

## Alias Discipline

- `paper_prod_qjl` is the primary two-stage preset name for new comparisons.
- `paper_prod` remains a paper-facing alias for compatibility.
- `high_compression` remains a compatibility alias and should not be treated as a separate paper-facing method.

## Scope

- Paper-facing does not mean product-supported beyond the current bounded Apple-MLX contract.
- Supported non-paper-facing means the branch is addressable in the repo and may have family-scoped runtime evidence, but it is outside the paper-facing story.
- Compatibility-only means the preset remains available for historical configs or comparisons, not as the primary algorithm narrative.
