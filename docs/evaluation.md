# TurboQuant Evaluation Guide

This document describes exploratory quality evaluation for TurboQuant. The thresholds here are local heuristics,
not certification gates and not production guarantees.

## Quick start

```python
from turboquant.config import TurboQuantConfig
from turboquant.eval import drift_report, memory_report, perplexity_report

cfg = TurboQuantConfig.from_preset("paper_prod")

ppl = perplexity_report(model, input_ids, turboquant_config=cfg, model_family="llama")
drift = drift_report(model, input_ids, turboquant_config=cfg, model_family="llama")
mem = memory_report(model, input_ids, turboquant_config=cfg, model_family="llama")
```

Use `paper_prod` when you want to evaluate the production-style QJL path. Use `paper_mse` when you want the MSE-only
reference path used by the current batch quality guardrail.

## Metrics

### Perplexity delta

`delta_ppl = tq_ppl - dense_ppl`

What it tells you:

- how far TurboQuant moved from the dense-cache baseline on a teacher-forcing evaluation
- whether a config change caused a large regression

Illustrative local heuristic:

- `delta_ppl < 0.5` is often a useful tuning target, but it is not a certification gate

### KL divergence

KL divergence compares the dense and compressed token distributions directly.

What it tells you:

- whether the compressed path is producing materially different logits
- whether a change caused obvious distributional drift even when perplexity stays acceptable

Illustrative local heuristic:

- `mean_kl < 0.01` is a useful small-scale tuning target, but it is not a certification gate

### Memory ratio

Memory ratio compares dense-cache bytes to TurboQuant-cache bytes.

What it tells you:

- whether your config is actually reducing memory on the workload you care about
- whether value quantization is enabled as expected

Treat analytic bit-accounting values and measured cache-byte totals as separate pieces of evidence.

### Inner-product bias

Inner-product bias compares the estimated compressed attention scores against the
true rotated-space dot products on a fixed synthetic workload.

What it tells you:

- whether the paper-facing two-stage `paper_prod_qjl` path behaves differently
- from the scalar-only `paper_mse` baseline
- whether the residual sketch is measurably changing signed error, absolute
- error, and error variance rather than only existing in code

Illustrative use:

- retain `inner_product_bias_summary.json`, `inner_product_bias_metrics.csv`,
- and `inner_product_bias_summary.md` from
- `benchmarks/runtime_cert/run_inner_product_bias_eval.py` as research metrics,
- not certification gates

Interpretation rule:

- use the JSON and CSV outputs to track metric movement across commits
- use the Markdown summary for human review and artifact handoff
- do not translate directional bias changes on one fixed workload into a proof
- of unbiasedness for the full two-stage path

## Recommended local workflow

1. Start with `TurboQuantConfig.from_preset("paper_prod")` for the production-style path.
2. Measure memory reduction with `memory_report(...)`.
3. Check drift with `drift_report(...)` on a short held-out sequence.
4. Check perplexity delta with `perplexity_report(...)`.
5. If you need a batch-quality reference, compare against `paper_mse`.
6. If you need a paper-facing score diagnostic, run the inner-product bias lane
   and compare `paper_prod_qjl` against `paper_mse` on the retained synthetic
   workload, then read the Markdown summary and CSV companion outputs together.
7. If you need a multi-point research comparison across several operating
   points, run the bit-budget sweep and compare its JSON, CSV, and Markdown
   outputs together.
8. If you need one consolidated KV report bundle, run the unified KV paper
   evaluation command and read its stage statuses by tier instead of treating
   all evidence as equally strong.

## Interpreting exploratory results

| Metric | Useful local heuristic | Typical response if worse |
|---|---|---|
| `delta_ppl` | `< 0.5` | raise `k_bits`, raise `v_bits`, or compare `paper_prod` against `paper_mse` |
| `mean_kl` | `< 0.01` | raise `k_bits`, increase `qjl_proj_dim`, or switch to `paper_mse` for diagnosis |
| memory ratio | `> 3x` at longer contexts | verify `v_enabled`, check value bit-widths, and inspect measured cache bytes |

These are exploratory heuristics only. They are not certification gates.

## How this differs from runtime certification

The current runtime certification workflow uses a batch teacher-forcing quality guardrail for the Llama scope.
That guardrail:

- uses `paper_mse`
- is intended to catch catastrophic regressions
- is not a general streaming-quality promise for every supported family

For release-facing evidence, rely on the certification artifacts and manifest rather than this document.
