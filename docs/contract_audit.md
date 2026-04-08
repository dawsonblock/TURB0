# Contract Audit

This audit records the actual TurboQuant support contract in the checked-in
tree and the evidence rules enforced by the repository. It does not assume
that a portable source snapshot contains a current Apple certification bundle.

## Canonical runtime path

- Single machine-readable authority: `turboquant/contract.json`
- Support-gated upgrade entry point: `turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)`
- Patched upstream decode hook: `mlx_lm.generate.generate_step(...)` delegates to `upgrade_cache_list(...)` when TurboQuant runtime arguments are provided
- Attention consumption path: `mlx_lm.models.base.scaled_dot_product_attention(...)` dispatches `TurboQuantKeysView` to `turboquant.runtime.attention.turboquant_streaming_attention(...)`
- Runtime allowlist gate: `turboquant.runtime.support.SUPPORTED_FAMILIES`, loaded from `turboquant/contract.json`

## Secondary and bypass surfaces

These surfaces exist in the source tree but are not peer supported runtime entry points:

- `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache` — internal direct adapter construction
- `integrations.mlx.upgrade.upgrade_cache_list(...)` — deprecated compatibility shim over the canonical runtime entry point
- `turboquant.eval.compare._collect_logits_compressed()` — internal eval helper that constructs TurboQuant caches directly

Direct adapter construction and eval helpers bypass the model-family support gate. The supported contract is the gated `upgrade_cache_list(...)` path; the deprecated `integrations.mlx.upgrade` shim lands back on that same path.

## Public claims retained

- Platform scope: `darwin-arm64` on Apple Silicon only
- Runtime scope: Python 3.9 through 3.11, MLX `>= 0.30.0` and `< 1.0.0`
- Allowlisted families: `llama` and `gemma`
- Paper-facing presets: `paper_mse` and `paper_prod` / `paper_prod_qjl`
- Evidence rule: source archives document workflow shape but do not prove a current PASS without an addressable workflow artifact, release evidence bundle, or pinned manifest digest

## Evidence rules present in this workspace

Workflow definitions:

- `.github/workflows/apple-runtime-cert.yml` — self-hosted Apple runtime certification workflow
- `.github/workflows/release.yml` — tagged release gate that requires Apple certification, both allowlisted families in scope, and the retained contract snapshot
- `turboquant/contract.json` — machine-readable source of truth for the
  source-archive evidence rule and the required release artifact set

Portable source snapshots are not expected to include
`artifacts/runtime-cert/` directories. Final release claims must instead point
to a published workflow artifact, release evidence bundle, or pinned manifest
digest from the tagged Apple-arm64 run.

## Evidence without product claims

When Apple certification is run, the resulting evidence bundle can contain
benchmark and exploratory detail such as paired dense-vs-TurboQuant JSON
outputs, `aggregate_runs.csv`, `certification_summary.json`, and optional
`events.jsonl`. Those files are useful evidence, but primary docs should not
convert them into timeless benchmark claims without pinned provenance.

## Claims deliberately removed or narrowed

- This repo does not ship a vendored `mlx_lm` tree, and upstream patch reachability is not described as blanket support.
- Direct adapter construction is no longer treated as a peer public runtime API.
- Historical benchmark tables were removed from primary docs unless they can be tied to addressable evidence.
- The exploratory real-model `paper_mse` quality tests are no longer presented as certification or product-proof gates.

## Residual constraints

- Evidence depth remains asymmetric: Llama is stronger; Gemma remains narrower overall because the conservative `paper_mse` batch quality guardrail remains Llama-scoped even though PolarQuant runtime and quality evidence now exist for Gemma.
- `legacy_topk` remains a compatibility branch, and `polarquant_exp` now works through the allowlisted runtime path with Llama and Gemma certification smoke stages plus family-scoped batch quality guardrails as a supported non-paper-facing branch.
- Family-scoped local PASS runs do not substitute for the tagged release workflow requirement that both allowlisted families appear in the manifest scope.
