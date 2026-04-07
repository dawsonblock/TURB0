# Benchmark Methodology

This document defines how benchmark numbers may be produced, reproduced, and published for TurboQuant.
It is not part of the certified product contract, and it intentionally avoids preserving historical tables
as if they were current facts.

## Core rule

Primary docs may publish benchmark numbers only when each published number maps to an addressable evidence bundle
or manifest digest plus exact provenance.

Required provenance fields:

- `artifact_uri_or_manifest_digest`
- `git_commit`
- `model_ids`
- `mlx_version`
- `hardware`
- `script`
- `args`

If a number cannot be tied to that provenance block, it should stay in a local lab notebook or artifact bundle,
not in the primary docs.

## What this means for the repo

- Historical benchmark snapshots have been removed from the primary docs.
- Benchmark commands remain documented here.
- Release evidence should carry the raw benchmark outputs, not just prose summaries.
- Source archives do not embed `artifacts/runtime-cert/<timestamp>/`; they document the workflow shape only.

## Benchmark categories

### Exploratory micro-benchmarks

These scripts live under `benchmarks/exploratory/` and are for local investigation, tuning, and profiling.

```bash
python benchmarks/exploratory/bench_memory_footprint.py
python benchmarks/exploratory/bench_dense_vs_turboquant.py
python benchmarks/exploratory/bench_decode_streaming.py
python benchmarks/exploratory/bench_decode_step.py
python benchmarks/exploratory/bench_k_encode.py
```

Use exploratory scripts when you want to understand behavior locally. Do not treat their outputs as release claims
unless you also record the provenance block above and attach or pin the resulting artifacts.

### Runtime-cert benchmark sweeps

These scripts live under `benchmarks/runtime_cert/` and are intended for structured paired dense-vs-TurboQuant runs.

```bash
python benchmarks/runtime_cert/run_dense_vs_tq.py \
    --model <hf-model-id> \
    --prompt-file benchmarks/runtime_cert/prompts/short.jsonl \
    --prompt-class short \
    --output-dir artifacts/runtime-cert/manual_run \
    --max-new-tokens 64 \
    --seed 42 \
    --mode both
```

For release-facing evidence, prefer the certification script so the benchmark outputs travel with `cert_manifest.json`,
`contract.json`, and the other required evidence files.

### Research-only score diagnostics

The runtime-cert artifact directory may also retain research-only score
diagnostics that are not release gates.

```bash
python benchmarks/runtime_cert/run_inner_product_bias_eval.py \
    --output-dir artifacts/runtime-cert/manual_run
```

This synthetic lane compares `paper_mse` against `paper_prod_qjl` on a fixed
attention-score workload and emits:

- `inner_product_bias_summary.json`
- `inner_product_bias_metrics.csv`
- `inner_product_bias_summary.md`

Use it to make the paper-facing scalar-only and two-stage paths directly
measurable. The JSON is the structured record, the CSV is the tabular metric
surface, and the Markdown file is the human-review summary. Do not present any
of them as proof of unbiasedness unless stronger retained evidence supports
that claim.

### Research-only bit-budget sweeps

The runtime-cert benchmark area also includes a synthetic bit-budget sweep
driver for research-only operating-point comparisons.

```bash
python benchmarks/runtime_cert/run_bit_budget_sweep.py \
    --output-dir artifacts/runtime-cert/manual_bit_budget_sweep
```

This driver emits:

- `bit_budget_sweep_summary.json`
- `bit_budget_sweep_metrics.csv`
- `bit_budget_sweep_summary.md`

Use it to compare measured compression, memory reduction, distortion,
inner-product bias, and cache-update latency across a stable preset and bit
matrix. The downstream-quality fields remain nullable in this synthetic sweep;
real-model quality evidence still belongs to the runtime-cert quality lanes.
See [docs/bit_budget_sweep.md](docs/bit_budget_sweep.md) for the schema and
interpretation rules.

## Reproduction rules

- Synchronize compute before timing.
- Record the exact script and invocation arguments.
- Record the exact MLX version and hardware.
- Record the exact model ids used.
- Keep the commit hash associated with the run.
- Preserve the raw output files in the evidence bundle.

## Publication checklist for a single number

Before a benchmark number appears in a primary doc, confirm all of the following:

1. There is an addressable workflow artifact, release evidence bundle, or pinned manifest digest for the run.
2. The run records the exact commit, model ids, MLX version, hardware, script, and args.
3. The raw files needed to re-audit the number are present in the evidence bundle.
4. The number is described with its scope and limitations rather than presented as a timeless fact.

If any of those are missing, keep the number out of the primary docs.
