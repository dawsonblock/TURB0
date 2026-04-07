# KV Paper Evaluation

This document defines the unified research-only KV evaluation command.

It gathers the repo's existing KV runtime, long-context, dense-versus-paper,
and optional real-model sweep surfaces into one report bundle without changing
the supported product contract.

## Driver

Run the command with:

```bash
source .venv-cert311/bin/activate
python benchmarks/runtime_cert/run_kv_paper_eval.py \
    --output-dir artifacts/runtime-cert/manual_kv_eval
```

That default run executes the `fast-check` tier only.

To include real-model heavier stages, add model ids and opt in explicitly:

```bash
source .venv-cert311/bin/activate
python benchmarks/runtime_cert/run_kv_paper_eval.py \
    --output-dir artifacts/runtime-cert/manual_kv_eval \
    --include-heavy-offline \
    --llama-model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --gemma-model mlx-community/gemma-2-2b-it-4bit \
  --include-gemma-quality-research
```

## Tiers

### Fast-check

This tier is intended to stay quick and reproducible.

It currently consolidates:

- Llama runtime smoke
- Gemma runtime smoke
- long-context stability
- dense versus `paper_mse` at about `2.75` average KV bpc
- dense versus `paper_mse` at about `3.75` average KV bpc

These checks often run against TinyModel defaults when real-model variables are
not set. They verify runtime-path continuity and numerical stability, not
real-model product-quality claims.

### Heavy-offline

This tier is optional and requires explicit real-model ids.

It currently consolidates:

- dense-versus-TurboQuant sweep artifacts from `run_dense_vs_tq.py`
- Llama `paper_mse` quality summaries from `run_quality_eval.py`
- optional Gemma `paper_mse` observational summaries from `run_quality_eval.py`

Gemma remains narrower here on purpose. The stronger `paper_mse` quality
guardrail remains Llama-scoped. When `--include-gemma-quality-research` is set,
the bundle records a separate Gemma observational tranche, but it stays marked
as research-only rather than being fabricated into symmetry.

## Output Files

The orchestrator emits:

- `kv_paper_eval_summary.json`
- `kv_paper_eval_summary.md`

The summary JSON records stage-level status, tier, commands, artifact paths,
and any measured benchmark or quality aggregates returned by the heavier stages.

## Interpretation

- A passing `fast-check` bundle means the runtime path and the lightweight
  paper-facing comparisons still work end to end.
- A missing heavy-offline stage is not a hidden pass; it should appear as
  `not_requested` or `not_configured` in the report.
- Heavy-offline results are still research evidence, not automatic product
  support expansion.
- A captured Gemma observational tranche records `paper_mse` metrics without
  promoting Gemma into a symmetric release guardrail.
- Read the unified bundle together with the narrower contract docs and
  certification artifacts.
