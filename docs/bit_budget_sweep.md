# Bit-Budget Sweep

This document defines the research-only bit-budget sweep surface.

It does not widen the supported product contract. The Apple-MLX release gate
and the contract-derived required release artifacts remain unchanged.

## Purpose

The bit-budget sweep is the paper-alignment lane for comparing several explicit
operating points across the stable preset taxonomy.

The default sweep is synthetic and reproducible. It is meant to answer a narrow
question:

- how compression, distortion, score-error metrics, and update latency move as
  the configured bit budget changes

It is not meant to replace real-model quality evidence.

## Driver

Run the sweep with:

```bash
source .venv-cert311/bin/activate
python benchmarks/runtime_cert/run_bit_budget_sweep.py \
    --output-dir artifacts/runtime-cert/manual_bit_budget_sweep
```

The default matrix sweeps:

- `paper_mse`
- `paper_prod_qjl`
- `polarquant_exp`

across `k_bits = 4, 3, 2` with fixed synthetic workloads.

## Output Files

The driver emits three files into the selected output directory:

- `bit_budget_sweep_summary.json`
- `bit_budget_sweep_metrics.csv`
- `bit_budget_sweep_summary.md`

## Report Schema

Each operating point records:

- configured preset and classification
- configured `k_bits`, `v_bits`, and group sizes
- measured effective bits per channel
- measured compression ratio
- memory footprint reduction
- key distortion metrics
- inner-product bias metrics
- dense versus TurboQuant cache-update latency
- downstream-quality placeholders

The downstream-quality fields are intentionally nullable in this synthetic
sweep. Real quality evidence remains available through the runtime-cert quality
artifacts and should not be inferred from the synthetic sweep alone.

## Interpretation

- Prefer measured compression ratio and measured memory reduction over paper-like
  intuition when reading the sweep.
- Treat bias movement as empirical evidence only; it is not a theorem claim.
- Treat latency overhead as a synthetic cache-update comparison, not a full
  model-serving latency promise.
- If downstream-quality fields are null, that means the sweep did not run a
  real-model long-context or teacher-forcing quality evaluation.

## Markdown Summary Template

Generated summaries should keep this structure:

```markdown
# Bit-Budget Sweep Summary

## Scope

## Workload

## Operating Points

## Honest Takeaways

## Companion Artifacts
```

That structure is stable on purpose so research outputs remain easy to compare
across commits without being confused for product-certification proof.
