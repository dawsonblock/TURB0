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
Only model families explicitly listed in `turboquant/runtime/support.py` are in the wired allowlist.
- **Llama-family** (Llama 2, Llama 3, Llama 3.1) — **wired, uncertified**
- **Gemma-family** (Gemma, Gemma 2) — **wired, uncertified**
- Other models (e.g., Qwen, Mistral, Phi, Falcon, Baichuan, Yi) may exist in the `mlx_lm` vendored directory but are considered **exploratory**, **vendored-only**, or **unsupported** unless added to the allowlist. Only `llama` and `gemma` families are in the wired allowlist.

## 4. Canonical Import Surfaces
To ensure long-term compatibility, users must only import from:
- `turboquant.*` (Core API)
- `turboquant.integrations.mlx.*` (MLX Integration)

Root-level `integrations/` are legacy compatibility shims and will be removed in a future release.

## 5. Experimental Features
- **Metal Kernels:** Custom Metal kernels (invoked via `TQ_USE_METAL=1`) are **experimental**. The default certified path uses the standard MLX Python/C++ boundary.
- **Exploratory Presets:** Any configuration not reachable via `TurboQuantConfig.from_preset()` is considered exploratory.

## 6. Runtime Certification

> **STATUS: NOT CERTIFIED.** No certification artifacts have been produced with real model weights.
> The quality evaluation script (`benchmarks/runtime_cert/run_quality_eval.py`) is implemented.
> Integration smoke tests exist but require `TQ_TEST_LLAMA_MODEL` or `TQ_TEST_GEMMA_MODEL` env
> variables pointing to real model weights to execute the memory-backed path.

"Full TurboQuant" status requires artifact-backed evidence generated via `make certify-apple-runtime`.
- Generic CI passes do not constitute runtime certification.
- Certification artifacts must include memory, latency, and quality benchmarks.
- Certification must begin with **Llama-family only** before Gemma is attempted.
