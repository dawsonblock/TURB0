<!-- Generated from turboquant/contract.json by scripts/render_support_contract.py. Do not edit by hand. -->
# TurboQuant Support Matrix

TurboQuant's narrow support boundary is generated from `turboquant/contract.json`.
Working trees may retain generated `artifacts/runtime-cert/` bundles for archaeology, but built wheels and source distributions do not ship those directories. No source or built snapshot proves a current PASS unless it is accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

## Algorithm Presets

| Preset | Classification | Canonical preset | Canonical algorithm | Residual | Effective K bpc (d=128) | Average KV bpc (d=128) | Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `paper_mse` | paper-facing | `paper_mse` | `paper_mse` | `none` | 3.25 | 3.75 | Paper-facing MSE stage. |
| `paper_prod_qjl` | paper-facing | `paper_prod_qjl` | `paper_prod_qjl` | `qjl` | 2.875 | 3.562 | Primary paper-facing two-stage preset using a 1-bit QJL residual. |
| `paper_prod (preset alias)` | paper-facing | `paper_prod_qjl` | `paper_prod_qjl` | `qjl` | 2.875 | 3.562 | Paper-facing production-style preset using a 1-bit QJL residual. |
| `polarquant_exp (supported non-paper-facing)` | supported non-paper-facing | `polarquant_exp` | `polarquant_exp` | `none` | 3.25 | 3.75 | Supported non-paper-facing PolarQuant branch with family-scoped runtime and quality certification. |

Paper-facing presets are `paper_mse`, `paper_prod_qjl`, and the paper-facing alias `paper_prod`. `polarquant_exp` remains the supported non-paper-facing branch.

## Exact deviations from the paper-facing story

- **Non-power-of-two Hadamard handling** — The implementation uses an exact Hadamard transform only for power-of-two head dimensions and a deterministic orthogonal fallback otherwise.
- **Legacy compatibility knobs** — residual_topk and block_tokens remain as narrow compatibility knobs, but compatibility presets are no longer part of the product contract.
- **Compatibility and non-paper-facing branches** — legacy_topk remains a compatibility branch, while polarquant_exp is now a supported non-paper-facing branch: PolarQuant works through the allowlisted upgrade_cache_list path, has Llama and Gemma certification runtime smoke stages, and family-scoped batch quality guardrails, but it remains outside the paper-facing preset story.
- **Monkey-patched upstream mlx_lm integration** — TurboQuant now patches upstream mlx_lm at import time instead of shipping a vendored fork, while keeping the allowlisted upgrade_cache_list gate as the canonical runtime path.

## Model Architecture Matrix

| Model family | Canonical support status | Evidence depth | Notes |
| :--- | :--- | :--- | :--- |
| Llama | Allowlisted via `upgrade_cache_list(...)` | stronger | The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Llama. Source archives alone do not prove a current PASS for Llama; use an addressable evidence bundle or pinned manifest digest. Coverage: real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, batch quality guardrail, long-context stability, dense-vs-TurboQuant benchmark sweeps. |
| Gemma | Allowlisted via `upgrade_cache_list(...)` | narrower | The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Gemma. Gemma coverage remains narrower overall because the conservative paper_mse batch quality guardrail remains Llama-scoped, even though PolarQuant runtime and quality evidence now exist for Gemma; source archives alone do not prove a current PASS. Coverage: real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, dense-vs-TurboQuant benchmark sweeps. |

## MLX Compatibility

- MLX >= 0.30.0 and < 1.0.0

## Hardware

- Apple Silicon (darwin-arm64)
