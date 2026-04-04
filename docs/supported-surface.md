<!-- Generated from turboquant/contract.json by scripts/render_support_contract.py. Do not edit by hand. -->
# Supported surface

TurboQuant's supported surface is generated from `turboquant/contract.json`.
Source archives document the certification workflow but do not prove a current PASS unless they are accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

## Supported slice

- darwin-arm64 on Apple Silicon
- Python 3.9 through 3.11 (recommended 3.11)
- MLX >= 0.30.0 and < 1.0.0
- Canonical runtime entry point: `upgrade_cache_list(...)`
- Research and local evaluation workflows only

## Model Support Matrix

| Model family | Support status | Evidence depth | Notes |
| :--- | :--- | :--- | :--- |
| Llama | Allowlisted | stronger | The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Llama. Source archives alone do not prove a current PASS for Llama; use an addressable evidence bundle or pinned manifest digest. |
| Gemma | Allowlisted | narrower | The release workflow is designed to produce addressable Apple-arm64 certification artifacts for Gemma. Gemma coverage is narrower than Llama because the batch quality guardrail remains Llama-scoped; source archives alone do not prove a current PASS. |

## Secondary surfaces

These surfaces exist, but they are not peer public runtime entry points:

| Surface | Status | Why it is secondary | Preferred path |
| :--- | :--- | :--- | :--- |
| `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache` | internal | bypasses the model-family allowlist | `turboquant.integrations.mlx.upgrade.upgrade_cache_list` |
| `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache` | compatibility shim | bypasses the model-family allowlist | `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache` |
| `mlx_lm.models.cache.KVCache._to_turboquant()` | private compatibility helper | bypasses the model-family allowlist | `turboquant.integrations.mlx.upgrade.upgrade_cache_list` |
| `mlx_lm.models.cache.KVCache.to_turboquant()` | deprecated public alias | bypasses the model-family allowlist | `turboquant.integrations.mlx.upgrade.upgrade_cache_list` |
| `turboquant.eval.compare._collect_logits_compressed()` | internal eval helper | constructs TurboQuantKCache directly for comparison harnesses | `turboquant.integrations.mlx.upgrade.upgrade_cache_list` |

## Release evidence contract

A release claim is only addressable when the workflow publishes or references:

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

## Benchmark publication rule

Primary docs may publish benchmark numbers only when each published number maps to an addressable evidence bundle or manifest digest plus exact commit, model ids, MLX version, hardware, script, and invocation arguments.

Required provenance fields:

- `artifact_uri_or_manifest_digest`
- `git_commit`
- `model_ids`
- `mlx_version`
- `hardware`
- `script`
- `args`
