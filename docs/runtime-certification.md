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
- Evidence depth is asymmetric: Llama is stronger; Gemma is narrower because the current batch quality guardrail remains Llama-scoped.

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

Artifacts are written under `artifacts/runtime-cert/<timestamp>/` during the run. Source archives do not
embed those generated directories. A release claim is only addressable when the workflow publishes or pins
that evidence as a workflow artifact, release evidence bundle, or manifest digest.

## Stage layout

The certification script runs the following stages:

1. Strict preflight
2. Cache upgrade roundtrip
3. Streaming attention equivalence
4. Llama smoke test
5. Gemma smoke test
6. Llama batch quality guardrail for short and medium prompts
7. Long-context stability
8. Dense-vs-TurboQuant benchmark sweeps
9. Metric aggregation
10. Contract snapshot (`contract.json`)

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
- `junit_gemma_smoke.xml`
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

- It currently runs only on the Llama scope.
- It uses the `paper_mse` preset rather than the production-style `paper_prod` path.
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

## Related docs

- [docs/product_contract.md](docs/product_contract.md)
- [docs/supported-surface.md](docs/supported-surface.md)
- [docs/support_matrix.md](docs/support_matrix.md)
- [docs/validation-local.md](docs/validation-local.md)