# Contract Audit

This audit records the actual TurboQuant support contract in the checked-in tree and the evidence that is addressable from this workspace.

## Canonical runtime path

- Single machine-readable authority: `turboquant/contract.json`
- Support-gated upgrade entry point: `turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)`
- Convenience wrapper: `mlx_lm.generate.maybe_turboquant_k_cache(...)`, which delegates to `upgrade_cache_list(...)`
- Attention consumption path: `mlx_lm.models.base.scaled_dot_product_attention(...)` dispatches `TurboQuantKeysView` to `turboquant.runtime.attention.turboquant_streaming_attention(...)`
- Runtime allowlist gate: `turboquant.runtime.support.SUPPORTED_FAMILIES`, loaded from `turboquant/contract.json`

## Secondary and bypass surfaces

These surfaces exist in the source tree but are not peer supported runtime entry points:

- `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache` — internal direct adapter construction
- `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache` — compatibility shim over the internal adapter
- `mlx_lm.models.cache.KVCache._to_turboquant()` — private compatibility helper
- `mlx_lm.models.cache.KVCache.to_turboquant()` — deprecated public alias
- `turboquant.eval.compare._collect_logits_compressed()` — internal eval helper that constructs TurboQuant caches directly

All of those surfaces bypass the model-family support gate. The supported contract is the gated `upgrade_cache_list(...)` path.

## Public claims retained

- Platform scope: `darwin-arm64` on Apple Silicon only
- Runtime scope: Python 3.9 through 3.11, MLX `>= 0.30.0` and `< 1.0.0`
- Allowlisted families: `llama` and `gemma`
- Paper-facing presets: `paper_mse` and `paper_prod` / `paper_prod_qjl`
- Evidence rule: source archives document workflow shape but do not prove a current PASS without an addressable workflow artifact, release evidence bundle, or pinned manifest digest

## Evidence sources present in this workspace

Workflow definitions:

- `.github/workflows/apple-runtime-cert.yml` — self-hosted Apple runtime certification workflow
- `.github/workflows/release.yml` — tagged release gate that requires Apple certification, both allowlisted families in scope, and the retained contract snapshot

Retained local evidence directories:

- `artifacts/runtime-cert/20260404_013136` — `PASS`, family-scoped to `llama`
- `artifacts/runtime-cert/20260404_013527` — `PASS`, family-scoped to `gemma`
- `artifacts/runtime-cert/20260404_015658` — `PASS`, combined `certification_scope.families=["gemma", "llama"]`

The combined retained run under `artifacts/runtime-cert/20260404_015658` contains the machine-readable and human-auditable evidence expected by the current release-facing contract, including:

- `contract.json`
- `cert_manifest.json`
- `preflight.json`
- `junit_cache_roundtrip.xml`
- `junit_attention_equiv.xml`
- `junit_llama_smoke.xml`
- `junit_gemma_smoke.xml`
- `junit_long_context.xml`
- `aggregate_runs.csv`
- `certification_summary.json`

## Evidence without product claims

The retained evidence directories also contain benchmark and exploratory detail such as paired dense-vs-TurboQuant JSON outputs, `aggregate_runs.csv`, `certification_summary.json`, and optional `events.jsonl`. Those files are kept as evidence, but primary docs no longer convert them into timeless benchmark claims without pinned provenance.

## Claims deliberately removed or narrowed

- The vendored `mlx_lm` tree is no longer described as blanket support for every model file it contains.
- Direct adapter construction is no longer treated as a peer public runtime API.
- Historical benchmark tables were removed from primary docs unless they can be tied to addressable evidence.
- The exploratory real-model `paper_mse` quality tests are no longer presented as certification or product-proof gates.

## Residual constraints

- Evidence depth remains asymmetric: Llama is stronger; Gemma is narrower because the batch quality guardrail remains Llama-scoped.
- Legacy compatibility branches (`legacy_topk`, `polarquant_exp`) remain in code but are outside the paper-facing supported story.
- Family-scoped local PASS runs do not substitute for the tagged release workflow requirement that both allowlisted families appear in the manifest scope.