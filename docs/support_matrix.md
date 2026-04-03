# TurboQuant Support Matrix

TurboQuant is an experimental KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation. Custom Metal kernels are experimental and not part of the default supported runtime.

Please refer to [supported-surface.md](supported-surface.md) for the canonical and complete supported surface details. This file is a matrix, not a broader support promise.

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

| Model Architecture | Runtime Verified | Notes |
|:---|:---:|:---|
| Llama | ⬜ | Wiring present in `mlx_lm/models/llama.py`; certification incomplete |
| Gemma | ⬜ | Wiring present in `mlx_lm/models/gemma.py`; certification incomplete |
| Qwen | ⬜ | Vendored-only in this repo; not supported by `upgrade_cache_list` |
| Mistral | ⬜ | Vendored-only in this repo; not supported by `upgrade_cache_list` |
| Phi | ⬜ | Vendored-only in this repo; not supported by `upgrade_cache_list` |
| All others | ⬜ | Unsupported; not in the certified allowlist. `upgrade_cache_list` raises `UnsupportedModelError` for these families. |

## MLX Compatibility

Tested against:

- MLX >= 0.30.0

## Hardware

- Apple Silicon (M1/M2/M3/M4) — Darwin arm64 natively supported.
