<!-- Generated from runtime-cert artifacts by scripts/render_benchmark_snapshot.py. Do not edit by hand. -->
# Benchmark Snapshot 20260407_014958

This is a dated benchmark snapshot, not a timeless product claim.
It summarizes the dense-vs-TurboQuant sweep retained in one runtime-cert evidence bundle.

## What This Helps With

TurboQuant helps when KV-cache memory pressure is the bottleneck on the Apple-Silicon MLX path.
In this run, peak benchmark memory dropped by 90.8% to 95.9% across the allowlisted Llama and Gemma sweeps.
That makes it useful for fitting longer prompts or reducing KV footprint on the supported runtime path.

It does not help raw decode throughput in the current uncompiled path.
In this same run, TurboQuant throughput was 93.4% to 95.8% lower than dense baselines.
If your bottleneck is tokens-per-second or per-token latency rather than memory, these numbers argue against using it as a speed optimization.

## Honest Takeaway

On this commit and hardware, TurboQuant behaves like a memory-saving tradeoff rather than a speedup.
The strongest honest claim from this snapshot is:

- It preserved the supported Apple runtime path for `gemma, llama` with a `PASS` manifest on `darwin-arm64`.
- It cut measured peak benchmark memory by roughly 90.8% to 95.9%.
- It reduced measured decode throughput by roughly 93.4% to 95.8%.
- It is therefore a memory-footprint tool first, not a throughput benchmark winner.

## Sweep Summary

| Model | Prompt class | Dense peak | TurboQuant peak | Memory reduction | Dense TPS | TurboQuant TPS | Throughput delta |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | `short` | 8.00 MiB | 0.36 MiB | 95.6% (22.5x smaller) | 191.07 | 12.03 | -93.7% |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | `medium` | 8.00 MiB | 0.43 MiB | 94.7% (18.7x smaller) | 189.39 | 12.48 | -93.4% |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | `long` | 22.40 MiB | 2.05 MiB | 90.8% (10.9x smaller) | 185.46 | 12.03 | -93.5% |
| `mlx-community/gemma-2-2b-it-4bit` | `short` | 26.00 MiB | 1.08 MiB | 95.9% (24.2x smaller) | 87.31 | 3.65 | -95.8% |
| `mlx-community/gemma-2-2b-it-4bit` | `medium` | 26.00 MiB | 1.32 MiB | 94.9% (19.6x smaller) | 84.09 | 3.62 | -95.7% |
| `mlx-community/gemma-2-2b-it-4bit` | `long` | 78.00 MiB | 6.82 MiB | 91.3% (11.4x smaller) | 87.02 | 3.81 | -95.6% |

## Inner-Product Bias Snapshot

This artifact also includes a research-only synthetic score comparison for the paper-facing scalar-only and two-stage paths.
It is evidence about current estimator behavior, not a release gate and not yet a proof of unbiasedness.

| Algorithm | Residual mode | Mean signed error | Mean abs error | Normalized mean bias | Normalized mean abs error | Error variance | 95th pct abs error |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper_mse` | `none` | -0.0332 | 1.8353 | -0.0036 | 0.1987 | 5.5274 | 4.6366 |
| `paper_prod_qjl` | `qjl` | -0.0861 | 2.0172 | -0.0093 | 0.2184 | 6.3815 | 4.7959 |

Bias snapshot takeaway:
The current `paper_prod_qjl` path differs from `paper_mse` by a normalized mean-abs-error delta of +0.0197 on this fixed synthetic workload.
That makes the two-stage path directly measurable, but it does not by itself justify an unbiasedness claim yet.

## Scope And Limits

- Commit under test: `31ae0d7`
- Models: `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/gemma-2-2b-it-4bit`
- Prompt classes: `long`, `medium`, `short`
- Decode length: `64` new tokens per paired run
- Benchmark mode: paired `dense` vs `turboquant` sweeps
- This snapshot does not prove a universal speedup, broad model-family support, or non-Apple portability.
- The quality stages in the certification bundle are guardrails against catastrophic regressions; they are not evidence that TurboQuant improves model quality.

## Provenance

- `artifact_uri_or_manifest_digest`: `artifacts/runtime-cert/20260407_014958.zip` and `sha256:32707125c8ee4f8a031205d4202fd330556ecd7f54fcca21906311b8dbad7ffd`
- `git_commit`: `31ae0d7`
- `model_ids`: `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/gemma-2-2b-it-4bit`
- `mlx_version`: `0.31.1`
- `hardware`: `macOS-26.2-arm64-arm-64bit`
- `script`: `bash scripts/certify_apple_runtime.sh`
- `args`: certification script invoked `benchmarks/runtime_cert/run_dense_vs_tq.py` for each model with `--prompt-file benchmarks/runtime_cert/prompts/{short,medium,long}.jsonl --max-new-tokens 64 --seed 42 --mode both`
- `extra_research_metrics`: optional `benchmarks/runtime_cert/run_inner_product_bias_eval.py` output when retained in the same artifact directory

## Addressable Evidence

- Local artifact directory: `/Users/dawsonblock/Downloads/TurboQuantX1-main/artifacts/runtime-cert/20260407_014958`
- Local portable artifact: `/Users/dawsonblock/Downloads/TurboQuantX1-main/artifacts/runtime-cert/20260407_014958.zip`
- Manifest: `/Users/dawsonblock/Downloads/TurboQuantX1-main/artifacts/runtime-cert/20260407_014958/cert_manifest.json`
- Hosted GitHub Actions evidence is still preferable for release-facing publication once a self-hosted Apple runner completes the queued workflow run.
