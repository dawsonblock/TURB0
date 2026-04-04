<!-- Generated from turboquant/contract.json by scripts/render_support_contract.py. Do not edit by hand. -->
# TurboQuant Product Contract

This document defines the narrow supported surface TurboQuant can honestly claim today.

TurboQuant supports one canonical runtime path via `upgrade_cache_list(...)`. A source archive documents that workflow, but it does not prove a current PASS without an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

## 1. Supported hardware and runtime

- Platform: `darwin-arm64`
- Hardware: Apple Silicon
- Python: 3.9 to 3.11 (recommended 3.11)
- MLX: >= 0.30.0 and < 1.0.0
- Scope: local Apple-Silicon MLX validation, not production deployment

## 2. Supported model families

- **Llama** — allowlisted; evidence depth is **stronger**. The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Llama. Source archives alone do not prove a current PASS for Llama; use an addressable evidence bundle or pinned manifest digest. Coverage: real-model smoke, experimental PolarQuant runtime smoke, experimental PolarQuant quality guardrail, batch quality guardrail, long-context stability, dense-vs-TurboQuant benchmark sweeps.
- **Gemma** — allowlisted; evidence depth is **narrower**. The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Gemma. Gemma coverage is narrower than Llama because the batch quality guardrail remains Llama-scoped; source archives alone do not prove a current PASS. Coverage: real-model smoke, experimental PolarQuant runtime smoke, dense-vs-TurboQuant benchmark sweeps.

## 3. Canonical and secondary surfaces

- Canonical runtime path: `upgrade_cache_list(...)`
- Secondary surfaces remain available only for compatibility or eval use:
  - `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache` (internal) — bypasses the model-family allowlist
  - `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache` (compatibility shim) — bypasses the model-family allowlist
  - `mlx_lm.models.cache.KVCache._to_turboquant()` (private compatibility helper) — bypasses the model-family allowlist
  - `mlx_lm.models.cache.KVCache.to_turboquant()` (deprecated public alias) — bypasses the model-family allowlist
  - `turboquant.eval.compare._collect_logits_compressed()` (internal eval helper) — constructs TurboQuantKCache directly for comparison harnesses

## 4. Paper-facing presets and exact deviations

Paper-facing presets are `paper_mse` and `paper_prod`/`paper_prod_qjl`. Legacy top-k presets remain compatibility paths, not the main algorithm story.

- **Non-power-of-two Hadamard handling** — The implementation uses an exact Hadamard transform only for power-of-two head dimensions and a deterministic orthogonal fallback otherwise.
- **Legacy compatibility knobs** — Legacy aliases, residual_topk, and block_tokens remain for compatibility, but they are not part of the paper-facing preset contract.
- **Vendored tree wider than support boundary** — The vendored mlx_lm tree contains many model files, but only the allowlisted families in this contract are supported by the canonical upgrade path.
- **Experimental branches outside paper-facing presets** — legacy_topk remains a compatibility branch and polarquant_exp remains an experimental runtime branch: PolarQuant now works through the allowlisted upgrade_cache_list path, has Llama and Gemma certification runtime smoke stages, and a Llama-scoped batch quality guardrail, but it is still outside the paper-facing preset story and formal supported product contract.

## 5. Release evidence and benchmarks

Source archives document the certification workflow but do not prove a current PASS unless they are accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

Primary docs may publish benchmark numbers only when each published number maps to an addressable evidence bundle or manifest digest plus exact commit, model ids, MLX version, hardware, script, and invocation arguments.

Required release artifacts:

- `contract.json`
- `cert_manifest.json`
- `preflight.json`
- `junit_cache_roundtrip.xml`
- `junit_attention_equiv.xml`
- `junit_llama_smoke.xml`
- `junit_polar_llama_runtime.xml`
- `quality_eval_polar_short_summary.json`
- `quality_eval_polar_medium_summary.json`
- `junit_gemma_smoke.xml`
- `junit_polar_gemma_runtime.xml`
- `junit_long_context.xml`
- `aggregate_runs.csv`
- `certification_summary.json`
