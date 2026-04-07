<!-- Generated from turboquant/contract.json by scripts/render_support_contract.py. Do not edit by hand. -->
# TurboQuant Product Contract

This document defines the narrow supported surface TurboQuant can honestly claim today.

TurboQuant supports one canonical runtime path via `upgrade_cache_list(...)`.

Working trees may retain generated `artifacts/runtime-cert/` bundles for archaeology, but built wheels and source distributions do not ship those directories. No source or built snapshot proves a current PASS unless it is accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

## 1. Supported hardware and runtime

- Platform: `darwin-arm64`
- Hardware: Apple Silicon
- Python: 3.9 to 3.11 (recommended 3.11)
- MLX: >= 0.30.0 and < 1.0.0
- Scope: local Apple-Silicon MLX validation, not production deployment

## 2. Supported model families

- **Llama** — allowlisted; evidence depth is **stronger**. The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Llama. Source archives alone do not prove a current PASS for Llama; use an addressable evidence bundle or pinned manifest digest. Coverage: real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, batch quality guardrail, long-context stability, dense-vs-TurboQuant benchmark sweeps.
- **Gemma** — allowlisted; evidence depth is **narrower**. The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Gemma. Gemma coverage remains narrower overall because the conservative paper_mse batch quality guardrail remains Llama-scoped, even though PolarQuant runtime and quality evidence now exist for Gemma; source archives alone do not prove a current PASS. Coverage: real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, dense-vs-TurboQuant benchmark sweeps.

## 3. Canonical and secondary surfaces

- Canonical runtime path: `upgrade_cache_list(...)`
- Secondary surfaces remain available only for compatibility or eval use:
  - `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache` (internal) — bypasses the model-family allowlist
  - `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache` (compatibility shim) — bypasses the model-family allowlist
  - `mlx_lm.models.cache.KVCache._to_turboquant()` (private compatibility helper) — bypasses the model-family allowlist
  - `mlx_lm.models.cache.KVCache.to_turboquant()` (deprecated public alias) — bypasses the model-family allowlist
  - `turboquant.eval.compare._collect_logits_compressed()` (internal eval helper) — constructs TurboQuantKCache directly for comparison harnesses

## 4. Paper-facing presets and exact deviations

Paper-facing presets are `paper_mse`, `paper_prod_qjl`, and the paper-facing alias `paper_prod`. `polarquant_exp` is the supported non-paper-facing branch. `legacy_topk`, `balanced`, `max_quality`, and `high_compression` remain compatibility-only surfaces rather than the main algorithm story.

Generated preset taxonomy: `docs/preset_modes.md`.

- **Non-power-of-two Hadamard handling** — The implementation uses an exact Hadamard transform only for power-of-two head dimensions and a deterministic orthogonal fallback otherwise.
- **Legacy compatibility knobs** — Legacy aliases, residual_topk, and block_tokens remain for compatibility, but they are not part of the paper-facing preset contract.
- **Vendored tree wider than support boundary** — The vendored mlx_lm tree contains many model files, but only the allowlisted families in this contract are supported by the canonical upgrade path.
- **Compatibility and non-paper-facing branches** — legacy_topk remains a compatibility branch, while polarquant_exp is now a supported non-paper-facing branch: PolarQuant works through the allowlisted upgrade_cache_list path, has Llama and Gemma certification runtime smoke stages, and family-scoped batch quality guardrails, but it remains outside the paper-facing preset story.

## 5. Release evidence and benchmarks

Working trees may retain generated `artifacts/runtime-cert/` bundles for archaeology, but built wheels and source distributions do not ship those directories. No source or built snapshot proves a current PASS unless it is accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

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
- `quality_eval_polar_gemma_short_summary.json`
- `quality_eval_polar_gemma_medium_summary.json`
- `junit_long_context.xml`
- `aggregate_runs.csv`
- `certification_summary.json`
