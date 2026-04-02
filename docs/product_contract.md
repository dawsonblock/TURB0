# TurboQuant Product Contract

This document defines the official supported surface and stability guarantees for TurboQuant.

## 1. Supported Hardware
TurboQuant is designed exclusively for **Apple Silicon** (M1, M2, M3, M4 families). 
- Non-Apple platforms are supported for packaging, linting, and static analysis only.
- Inference via MLX is not supported or certified on non-Apple hardware.

## 2. Supported Runtime
The canonical runtime is the **local MLX runtime** on macOS.
- Deployment via remote inference servers or non-macOS environments is currently out of scope.

## 3. Supported Model Families
Only model families explicitly listed in `turboquant/runtime/support.py` are certified for production use.
- **Llama-family** (Llama 2, Llama 3, Llama 3.1)
- **Gemma-family** (Gemma, Gemma 2)
- Other models (e.g., Qwen, Mistral, Phi, Falcon, Baichuan, Yi) may exist in the `mlx_lm` vendored directory but are considered **exploratory**, **vendored-only**, or **unsupported** unless added to the allowlist. Only `llama` and `gemma` families are officially supported.

## 4. Canonical Import Surfaces
To ensure long-term compatibility, users must only import from:
- `turboquant.*` (Core API)
- `turboquant.integrations.mlx.*` (MLX Integration)

Root-level `integrations/` are legacy compatibility shims and will be removed in a future release.

## 5. Experimental Features
- **Metal Kernels:** Custom Metal kernels (invoked via `TQ_USE_METAL=1`) are **experimental**. The default certified path uses the standard MLX Python/C++ boundary.
- **Exploratory Presets:** Any configuration not defined in `turboquant/presets.py` is considered exploratory.

## 6. Runtime Certification
"Full TurboQuant" status requires artifact-backed evidence generated via `make certify-apple-runtime`. 
- Generic CI passes do not constitute runtime certification.
- Certification artifacts must include memory, latency, and quality benchmarks for at least one Llama and one Gemma model.
