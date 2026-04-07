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
- Generated benchmark and family evidence indexes live in `docs/benchmark_index.md` and `docs/family_evidence_matrix.md` so the research lane stays explicit without becoming release truth.

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

### Unified KV paper evaluation bundle

The runtime-cert benchmark area also includes a unified research-only KV report
driver that consolidates fast runtime-path checks and optional heavy real-model
stages.

```bash
python benchmarks/runtime_cert/run_kv_paper_eval.py \
    --output-dir artifacts/runtime-cert/manual_kv_eval
```

By default this runs only the `fast-check` tier. Add
`--include-heavy-offline` plus model ids to run the heavier real-model stages.

It emits:

- `kv_paper_eval_summary.json`
- `kv_paper_eval_summary.md`

Use it when you want one report bundle that states clearly which KV results are
lightweight fast checks and which are heavier offline research evaluations.
See [docs/kv_paper_eval.md](docs/kv_paper_eval.md) for the tier definitions and
interpretation rules.

If you want those retained research summaries to appear in a dated benchmark
history snapshot, you can pass them to the snapshot renderer explicitly:

```bash
python scripts/render_benchmark_snapshot.py \
    --artifact-dir artifacts/runtime-cert/<certification-timestamp> \
    --kv-paper-eval-summary artifacts/runtime-cert/<kv-bundle>/kv_paper_eval_summary.json \
    --vector-search-summary artifacts/runtime-cert/<vector-search-bundle>/vector_search_summary.json
```

### Vector-search research lane

The repo also includes a bundled mini vector-search evaluation path under
`benchmarks/vector_search/`.

```bash
python benchmarks/vector_search/run_vector_search_eval.py \
    --output-dir artifacts/runtime-cert/manual_vector_search
```

It emits:

- `vector_search_summary.json`
- `vector_search_metrics.csv`
- `vector_search_summary.md`

Use it to compare dense retrieval against the current paper-facing and
non-paper-facing preset surfaces on a bundled mini dataset. This lane is
research-only and should not be described as part of the supported runtime
product surface. See [docs/vector_search.md](docs/vector_search.md).

For larger retrieval evidence, the same driver supports an explicit public
corpus download lane behind `--download-public-corpus` so the repo does not
ship third-party data by default.

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
