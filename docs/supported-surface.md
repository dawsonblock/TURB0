# Supported surface

TurboQuant is an experimental KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation for selected Llama-family and Gemma-family models. The canonical upgrade entry point is `upgrade_cache_list(...)` inside the `mlx_lm` generation flow. Custom Metal kernels are experimental and not part of the default supported runtime.

This repository does **not** claim broad `mlx_lm` model coverage. The codebase vendors a large upstream tree, but vendoring upstream code is not support. The TurboQuant-specific attention path is only wired and discussed for a narrow slice.

## Supported slice

What this repository can honestly claim to support:

- Apple Silicon Macs
- Python 3.9+
- MLX runtime installed locally
- Research and local evaluation workflows
- TurboQuant core package: `turboquant/*`
- `mlx_lm` cache-upgrade path via `upgrade_cache_list(...)` for allowlisted families
- `KVCompressor` as a compatibility alias for `TurboQuantKVCache`
- Llama-family integration path
- Gemma-family integration path

## Model Support Matrix

| Model Architecture | Explicit Integration Tested | Support Status | Notes |
| :--- | :--- | :--- | :--- |
| Llama | Yes | **Wired, uncertified** | Smoke test wired (`test_llama_runtime_smoke.py`); set `TQ_TEST_LLAMA_MODEL` to activate. Runtime certification not yet completed. |
| Gemma | Yes | **Wired, uncertified** | Smoke test wired (`test_gemma_runtime_smoke.py`); set `TQ_TEST_GEMMA_MODEL` to activate (run Llama first). Runtime certification not yet completed. |
| Qwen | No | Unsupported | Not in the allowlist; `upgrade_cache_list` rejects it. |
| Mistral | No | Unsupported | Not in the allowlist; `upgrade_cache_list` rejects it. |
| Phi | No | Unsupported | Provided via upstream sync only. |
| &lt;All Others&gt; | No | Unsupported | Rejected by `upgrade_cache_list`; vendored for structural scaffolding. |

## Secondary surfaces

These exist in the repo but are not the supported public runtime entry points:

- Direct `TurboQuantKCache(...)` construction for eval and compatibility helpers
- Legacy `maybe_turboquant_attention(...)` helper paths outside the main `base.py` SDPA type-guard
- `turboquant.runtime.events.EventLog` JSONL persistence for certification artifacts

## Not claimed

What is **not** claimed by the current repository state:

- Public CI runtime certification of MLX-backed generation
- Production SLOs
- Broad compatibility across every model in the vendored `mlx_lm/models/` tree
- Fused Metal kernels for encoding or decoding (Metal kernel integration remains experimental and is not the default supported path)
- Large-scale perplexity validation
- Generic Linux or Windows runtime support
- Production readiness

## Validation boundary

Two validation layers exist:

1. **Public static checks**
   - packaging metadata
   - source-tree integrity
   - syntax compilation
2. **Local Apple Silicon checks**
   - MLX install
   - unit and integration tests
   - structural proof tests (`make test-structural`, `make test-path-proof`)
   - model smoke runs (`make test-smoke-llama`, `make test-smoke-gemma`, `make test-long-context`)
   - manual memory and latency comparison

Use `scripts/preflight.py` for the first layer and `scripts/validate_apple_silicon.sh` for the second.
