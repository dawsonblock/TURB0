# Runtime Certification

This document defines the narrow Apple-Silicon MLX certification workflow for TurboQuant.
It describes how a release workflow is expected to produce evidence. It does not claim that a
source archive alone proves a current PASS.

## Current contract

- Supported platform: `darwin-arm64` on Apple Silicon.
- Supported Python range: 3.9 through 3.11, with 3.11 recommended.
- Supported MLX range: `>= 0.30.0` and `< 1.0.0`.
- Canonical runtime path: `upgrade_cache_list(...)` inside the `mlx_lm` decode flow.
- Supported families: `llama` and `gemma`.
- Evidence depth is asymmetric: Llama is stronger; Gemma remains narrower overall because the conservative `paper_mse` batch quality guardrail remains Llama-scoped even though PolarQuant runtime and quality gates now run on both families, and the unified KV bundle only records Gemma `paper_mse` as a research-only observational tranche when explicitly requested.

The machine-readable source of truth is `turboquant/contract.json`.

## What certification does and does not mean

Passing this workflow means only that the narrow Apple-Silicon MLX runtime path works for the
allowlisted families with generated evidence describing the run.

It does not certify:

- production readiness
- Linux or Windows
- CUDA or ROCm
- the full vendored `mlx_lm` tree
- custom Metal kernels
- distributed inference
- training or fine-tuning

## Interpreter bootstrap

`./scripts/certify_apple_runtime.sh` prefers `python3.11`, then `python3.10`, then `python3.9`,
and only then falls back to `python3`.

If you run the script inside an existing environment, keep that environment on Python 3.9-3.11.

## Environment

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"
./scripts/certify_apple_runtime.sh
```

Artifacts are written under `artifacts/runtime-cert/<timestamp>/` during the run. This working tree may retain
previously generated bundles for archaeology, but built wheels and source distributions do not ship those
generated directories. A release claim is only addressable when the workflow publishes or pins that evidence
as a workflow artifact, release evidence bundle, or manifest digest.

## Stage layout

The certification script runs the following stages:

1. Strict preflight
2. Cache upgrade roundtrip
3. Streaming attention equivalence
4. Llama smoke test
5. PolarQuant runtime smoke for Llama
6. Gemma smoke test
7. PolarQuant runtime smoke for Gemma
8. Llama batch quality guardrail for short and medium prompts (`paper_mse`)
9. Llama PolarQuant batch quality guardrail for short and medium prompts (`polarquant_exp`)
10. Gemma PolarQuant batch quality guardrail for short and medium prompts (`polarquant_exp`)
11. Long-context stability
12. Dense-vs-TurboQuant benchmark sweeps
13. Metric aggregation
14. Contract snapshot (`contract.json`)

`contract.json` is written into the artifact directory so the evidence bundle carries both the
run result and the exact contract the run was meant to satisfy.

## Required release evidence

The generated evidence directory is expected to contain at least:

- `contract.json`
- `cert_manifest.json`
- `preflight.json`
- `junit_cache_roundtrip.xml`
- `junit_attention_equiv.xml`
- `junit_llama_smoke.xml`
- `junit_polar_llama_runtime.xml`
- `junit_gemma_smoke.xml`
- `junit_polar_gemma_runtime.xml`
- `quality_eval_polar_short_summary.json`
- `quality_eval_polar_medium_summary.json`
- `quality_eval_polar_gemma_short_summary.json`
- `quality_eval_polar_gemma_medium_summary.json`
- `junit_long_context.xml`
- `aggregate_runs.csv`
- `certification_summary.json`

Optional persistence outputs may also exist:

- `events.jsonl` when a helper explicitly records runtime events through `EventLog`
- per-run benchmark JSON files such as `*_dense.json` and `*_turboquant.json`

The canonical decode path does not automatically persist `events.jsonl`; event persistence remains an
explicit certification or instrumentation surface.

## Manifest semantics

`cert_manifest.json` is the machine-readable pass or fail record for a run.

Important fields:

- `result`
- `platform`
- `certification_scope.families`
- stage counts and failure counts

The manifest is family-scoped. A run may be valid for only `llama`, only `gemma`, or both, depending on which
real-model families were in scope. Final release publication is stricter: the tagged workflow must validate a PASS
manifest that includes both `llama` and `gemma` in `certification_scope.families`.

## Quality-stage interpretation

The quality stage is a batch teacher-forcing guardrail, not a streaming-certification claim.

- The conservative `paper_mse` guardrail remains Llama-scoped.
- The supported non-paper-facing `polarquant_exp` guardrail now runs on both Llama and Gemma.
- It is designed to catch catastrophic regressions such as KV corruption, NaN propagation, or severe numerical drift.

Do not present this stage as proof of streaming decode quality for every supported family.

## Release gate expectations

For a tagged publish to be technically credible:

- the self-hosted Apple-Silicon workflow must run the certification script in that same workflow
- the workflow must upload or otherwise reference the generated evidence directory
- the manifest must record `result: PASS`
- the manifest must include both allowlisted families for the final release gate

If the self-hosted `macOS` `ARM64` runner pool is offline, the release should stay blocked rather than reusing a
previous local artifact or making a manual judgment call.

## Workflow operator runbook

To close the remaining release-proof gap from a clean repo snapshot:

1. Ensure an online GitHub Actions runner labeled `self-hosted`, `macOS`, `ARM64` is available.
2. Ensure both `TQ_TEST_LLAMA_MODEL` and `TQ_TEST_GEMMA_MODEL` secrets are configured.
3. Trigger `.github/workflows/apple-runtime-cert.yml` on a matching push to `main` or by manual dispatch with `run_model_stages` enabled.
4. Confirm `structural` passes on `macos-14` and `full-certification` passes on the self-hosted Apple runner.
5. Download the uploaded `runtime-cert-<sha>` artifact from that workflow run.
6. Validate the downloaded artifact with:

```bash
python scripts/verify_runtime_cert_artifact.py path/to/runtime-cert-<sha>.zip
```

The verifier accepts either the downloaded zip or an extracted artifact directory. It fails unless:

- `cert_manifest.json` records `result: PASS`
- `cert_manifest.json` records `platform: darwin-arm64`
- `certification_scope.families` is exactly `llama` and `gemma`
- the retained `contract.json` matches `turboquant/contract.json`
- every contract-driven required release artifact is present

Keep the uploaded artifact or a pinned manifest digest as the addressable release-proof citation. A local `artifacts/runtime-cert/` directory is useful operationally, but it is not a portable proof source by itself.

## Related docs

- [docs/product_contract.md](docs/product_contract.md)
- [docs/supported-surface.md](docs/supported-surface.md)
- [docs/support_matrix.md](docs/support_matrix.md)
- [docs/validation-local.md](docs/validation-local.md)
