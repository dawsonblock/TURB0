<!-- Generated from turboquant/contract.json by scripts/render_support_contract.py. Do not edit by hand. -->
# TurboQuant Support Matrix

TurboQuant's narrow support boundary is generated from `turboquant/contract.json`.
Working trees may retain generated `artifacts/runtime-cert/` bundles for archaeology, but built wheels and source distributions do not ship those directories. No source or built snapshot proves a current PASS unless it is accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

## Algorithm Presets

| Preset | Canonical algorithm | Residual | Effective K bpc (d=128) | Average KV bpc (d=128) | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `paper_mse` | `paper_mse` | `none` | 3.25 | 3.75 | Paper-facing MSE stage. |
| `paper_prod (preset alias)` | `paper_prod_qjl` | `qjl` | 2.875 | 3.562 | Paper-facing production-style preset using a 1-bit QJL residual. |
| `high_compression (legacy alias)` | `paper_prod_qjl` | `qjl` | 2.875 | 3.562 | Legacy convenience alias for the QJL production-style preset. |
| `balanced (legacy)` | `legacy_topk` | `topk` | legacy / compatibility-only | legacy / compatibility-only | Legacy top-k compatibility preset; not part of the paper-facing contract. |
| `max_quality (legacy)` | `legacy_topk` | `topk` | legacy / compatibility-only | legacy / compatibility-only | Legacy top-k compatibility preset; not part of the paper-facing contract. |
| `polarquant_exp (supported non-paper-facing)` | `polarquant_exp` | `none` | 3.25 | 3.75 | Supported non-paper-facing PolarQuant branch with family-scoped runtime and quality certification. |

Paper-facing presets are `paper_mse` and `paper_prod` (the `paper_prod_qjl` algorithm family). Legacy top-k presets remain available only as compatibility surfaces.

## Exact deviations from the paper-facing story

- **Non-power-of-two Hadamard handling** — The implementation uses an exact Hadamard transform only for power-of-two head dimensions and a deterministic orthogonal fallback otherwise.
- **Legacy compatibility knobs** — Legacy aliases, residual_topk, and block_tokens remain for compatibility, but they are not part of the paper-facing preset contract.
- **Vendored tree wider than support boundary** — The vendored mlx_lm tree contains many model files, but only the allowlisted families in this contract are supported by the canonical upgrade path.
- **Compatibility and non-paper-facing branches** — legacy_topk remains a compatibility branch, while polarquant_exp is now a supported non-paper-facing branch: PolarQuant works through the allowlisted upgrade_cache_list path, has Llama and Gemma certification runtime smoke stages, and family-scoped batch quality guardrails, but it remains outside the paper-facing preset story.

## Model Architecture Matrix

| Model family | Canonical support status | Evidence depth | Notes |
| :--- | :--- | :--- | :--- |
| Llama | Allowlisted via `upgrade_cache_list(...)` | stronger | The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Llama. Source archives alone do not prove a current PASS for Llama; use an addressable evidence bundle or pinned manifest digest. Coverage: real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, batch quality guardrail, long-context stability, dense-vs-TurboQuant benchmark sweeps. |
| Gemma | Allowlisted via `upgrade_cache_list(...)` | narrower | The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Gemma. Gemma coverage remains narrower overall because the conservative paper_mse batch quality guardrail remains Llama-scoped, even though PolarQuant runtime and quality evidence now exist for Gemma; source archives alone do not prove a current PASS. Coverage: real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, dense-vs-TurboQuant benchmark sweeps. |

## MLX Compatibility

- MLX >= 0.30.0 and < 1.0.0

## Hardware

- Apple Silicon (darwin-arm64)
