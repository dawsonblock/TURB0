# Runtime Certification

> **STATUS: RETAINED ARTIFACT-BACKED PASS ARTIFACTS EXIST FOR BOTH ALLOWLISTED FAMILIES.**
> Apple-arm64 PASS manifests now exist at `artifacts/runtime-cert/20260404_013136`
> for `llama` and `artifacts/runtime-cert/20260404_013527` for `gemma`.
> A retained combined release-equivalent PASS manifest also exists at
> `artifacts/runtime-cert/20260404_015658` with both families in scope.
> The Llama artifact covers real-model smoke, batch quality guardrail,
> long-context stability, and dense-vs-TQ benchmark sweeps on the canonical path.
> The Gemma artifact covers real-model smoke and dense-vs-TQ benchmark sweeps on the
> canonical path; the current batch quality guardrail remains Llama-scoped.
> Release must stay blocked unless a tagged run produces an equivalent PASS
> `cert_manifest.json`.

## Purpose

This document describes the **narrow** Apple-Silicon MLX runtime
certification surface for TurboQuant v0.2.2.

Passing this certification means **only**:

> The TurboQuant compressed KV-cache path works for the supported
> Apple-MLX runtime on selected Llama-family and Gemma-family models,
> with reproducible artifacts, bounded quality loss, and measurable
> memory benefit.

It does **not** certify production readiness, cross-platform support,
all model families, custom Metal kernels, or distributed inference.

---

## Supported certification surface

| Dimension         | Value                                    |
| ----------------- | ---------------------------------------- |
| Hardware          | Apple Silicon Mac (arm64)                |
| OS                | macOS (version recorded per run)         |
| Python            | 3.11 recommended; 3.9–3.11 supported. The certification script bootstrap prefers `python3.11`, then `python3.10`, then `python3.9`. |
| MLX               | ≥ 0.30.0 (exact version recorded)       |
| TurboQuant        | 0.2.2 (commit hash recorded)            |
| Llama model       | set via `TQ_TEST_LLAMA_MODEL` env var    |
| Gemma model       | set via `TQ_TEST_GEMMA_MODEL` env var    |
| Modes             | dense baseline, TurboQuant enabled       |
| Prompt classes    | short (5), medium (5), long (5)          |

## Unsupported surface

- Linux / Windows
- CUDA / ROCm
- General-purpose mlx_lm compatibility
- All model families
- Custom Metal kernel runtime
- Production readiness
- Distributed inference
- Training / fine-tuning

---

## Required environment

```text
macOS on Apple Silicon (M1/M2/M3/M4)
Python 3.11 (recommended; 3.9–3.11 supported)
A clean virtual environment
pip install -e '.[apple,test]'
```

If `./scripts/certify_apple_runtime.sh` bootstraps its own environment, it
prefers `python3.11`, then `python3.10`, then `python3.9`. If you run the
script inside an existing environment, keep that environment on Python 3.9–3.11.

## Environment variables

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"
```

Use small quantized models to keep certification runs fast.

---

## Exact command

```bash
./scripts/certify_apple_runtime.sh
```

This single command runs the full certification pipeline. Artifacts are
written to `artifacts/runtime-cert/<timestamp>/`.

For tagged releases, the self-hosted Apple job in `release.yml` runs this exact
command and validates the generated `cert_manifest.json`. PyPI publish stays blocked
unless that manifest exists and records `result: "PASS"` on `darwin-arm64`.
Tagged release publish also requires both `llama` and `gemma` to be present in
`certification_scope.families` for that manifest.

`cert_manifest.json` records `certification_scope.families` for the real-model families
selected in that run. A PASS can therefore be family-scoped while certification widens
(for example, `llama` first), but at least one real-model family must be in scope and
every in-scope stage must pass. Unselected family stages are recorded as out of scope,
not as skipped certification failures.

## Current retained PASS artifacts

- `artifacts/runtime-cert/20260404_013136/` — `certification_scope.families=["llama"]`; real-model smoke, batch quality guardrail, long-context stability, and dense-vs-TQ benchmark sweeps recorded on the canonical path.
- `artifacts/runtime-cert/20260404_013527/` — `certification_scope.families=["gemma"]`; real-model smoke and dense-vs-TQ benchmark sweeps recorded on the canonical path. The current batch quality guardrail remains Llama-scoped.
- `artifacts/runtime-cert/20260404_015658/` — `certification_scope.families=["gemma", "llama"]`; retained combined release-equivalent PASS artifact matching the tagged publish workflow requirement.

---

## Certification stages

| # | Stage                          | Tool                                                      |
| - | ------------------------------ | --------------------------------------------------------- |
| 1 | Strict preflight               | `python scripts/preflight.py --strict --json`             |
| 2 | Cache upgrade roundtrip        | `pytest tests/integration_mlx/test_cache_upgrade_roundtrip.py` |
| 3 | Streaming attention equivalence | `pytest tests/integration_mlx/test_streaming_attention_equivalence.py` |
| 4 | Llama smoke test               | `pytest tests/integration_mlx/test_llama_runtime_smoke.py` |
| 5 | Gemma smoke test               | `pytest tests/integration_mlx/test_gemma_runtime_smoke.py` |
| 6 | Long-context stability         | `pytest tests/integration_mlx/test_long_context_stability.py` |
| 7 | Quality evaluation             | `run_quality_eval.py` (short + medium, Llama)            |
| 8 | Dense vs TQ benchmarks         | `run_dense_vs_tq.py` × 3 prompt classes × 2 models       |
| 9 | Metric aggregation             | `collect_metrics.py`                                      |

---

## Artifacts produced

After a full run, `artifacts/runtime-cert/<timestamp>/` contains:

| File                             | Description                                |
| -------------------------------- | ------------------------------------------ |
| `cert_manifest.json`             | Machine-readable PASS/FAIL manifest used by the release gate |
| `preflight.json`                 | Machine-readable preflight result          |
| `junit_cache_roundtrip.xml`      | Cache roundtrip test results               |
| `junit_attention_equiv.xml`      | Attention equivalence test results         |
| `junit_llama_smoke.xml`          | Llama smoke test results                   |
| `junit_gemma_smoke.xml`          | Gemma smoke test results                   |
| `junit_long_context.xml`         | Long-context stability test results        |
| `*_dense.json`                   | Raw per-run dense benchmark results        |
| `*_turboquant.json`              | Raw per-run TurboQuant benchmark results   |
| `events.jsonl`                   | Optional persisted upgrade/failure events when a certification helper explicitly records them |
| `aggregate_runs.csv`             | All runs in tabular form                   |
| `certification_summary.json`     | Pass/fail rollup with memory/speed deltas  |

---

## Thresholds and pass/fail rules

A certification run **passes** only if all of the following are true:

### Scope

- At least one real-model family is selected in `certification_scope.families`
- Unselected family stages remain out of scope; they do not count as PASS, but they do not invalidate a family-scoped artifact either
- Tagged release publish requires both allowlisted families in `certification_scope.families`

### Structural

- Zero cache upgrade failures
- Zero state restore failures
- Zero sequence-offset mismatches

### Numerical (attention equivalence)

| Metric                | Threshold            |
| --------------------- | -------------------- |
| Cosine similarity     | ≥ 0.960              |
| Mean absolute error   | ≤ 0.06               |
| Max absolute error    | ≤ 0.25               |

Thresholds frozen after pilot run on Apple Silicon (M-series).
Observed cosine ~0.97 for 3-bit K + 4-bit V with Hadamard rotation.

### Runtime

- Zero crashes on supported models
- Zero empty generations in smoke tests
- No silent dense fallback when TurboQuant is requested

### Quality (batch guardrail, not streaming certification)

- 100% of prompts complete without crash
- No catastrophic degeneration on any prompt
- `run_quality_eval.py` uses the `paper_mse` preset for this batch teacher-forcing check
- Prompts shorter than 32 tokens are skipped for this gate
- Mean perplexity delta (TQ - dense) ≤ 20.0
- Mean KL divergence ≤ 5.0

### Performance

- Measurable memory reduction in TurboQuant long-context mode (≥ 25.0%)
- No evidence of catastrophic decode slowdown (degradation ≤ -99.0%)

The speed gate is intentionally loose because the default certification path is
the uncompiled Python/MLX streaming implementation. It is a catastrophic-failure
guard, not a throughput promise.

---

## Threshold freeze process

1. Change certification thresholds only when the underlying script or runtime behavior changes.
2. Update this document and the static contract tests in the same change.
3. Re-run `./scripts/certify_apple_runtime.sh` and inspect the saved artifacts.
4. Treat the saved artifacts, not this prose, as the authoritative pass/fail record.
