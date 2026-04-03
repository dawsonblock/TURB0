# TurboQuant Support Matrix

TurboQuant is an experimental KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation through the allowlisted `upgrade_cache_list(...)` path. Custom Metal kernels are experimental and not part of the default supported runtime.

This table records support status only. Mechanical routing in vendored model code does not widen support.

Please refer to [supported-surface.md](supported-surface.md) for the canonical and complete supported surface details. This file is a status matrix, not a broader support promise.

## Algorithm Presets

| Preset | Algorithm | K bits | V bits | Residual | Effective K bpc (d=128) |
|:---|:---|:---:|:---:|:---|:---:|
| `paper_mse` | `turboquant_mse` | 3 | 4 | none | 3.25 |
| `paper_prod` | `turboquant_prod` | 3 | 4 | 1-bit QJL | ~3.75 |
| `high_compression` (legacy) | `turboquant_prod` | 3 | 4 | QJL | ~3.75 |
| `balanced` (legacy) | `turboquant_prod` | 4 | 4 | QJL | ~4.5 |
| `max_quality` (legacy) | `turboquant_prod` | 4 | 8 | QJL | ~4.5 |

**Paper-faithful presets** (`paper_mse`, `paper_prod`) use:
- Hadamard rotation (`rotation="hadamard"`)
- Lloyd-Max scalar quantiser with Gaussian centroids
- `turboquant_mse`: rotate → Lloyd-Max scalar quant (no residual)
- `turboquant_prod`: MSE stage + 1-bit QJL residual for unbiased inner-product estimation

## Model Architecture Matrix

Only allowlisted families are eligible for the canonical runtime path today. Wiring in the vendored tree is not the same as support.

| Model Architecture | Runtime Verified | Notes |
|:---|:---:|:---|
| Llama | ⬜ | Allowlisted and wired; current status is **wired, uncertified** until Apple-Silicon runtime artifacts exist |
| Gemma | ⬜ | Allowlisted and wired; current status is **wired, uncertified** until Apple-Silicon runtime artifacts exist |
| Qwen | ⬜ | Vendored-only in this repo; unsupported by `upgrade_cache_list(...)` |
| Mistral | ⬜ | Vendored-only in this repo; unsupported by `upgrade_cache_list(...)` |
| Phi | ⬜ | Vendored-only in this repo; unsupported by `upgrade_cache_list(...)` |
| All others | ⬜ | Unsupported; not in the allowlist. `upgrade_cache_list(...)` raises `UnsupportedModelError` for these families. |

## MLX Compatibility

Tested against:

- MLX >= 0.30.0

## Hardware

- Apple Silicon (M1/M2/M3/M4) — Darwin arm64 natively supported.
