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

## Recommended local workflow

1. Start with `TurboQuantConfig.from_preset("paper_prod")` for the production-style path.
2. Measure memory reduction with `memory_report(...)`.
3. Check drift with `drift_report(...)` on a short held-out sequence.
4. Check perplexity delta with `perplexity_report(...)`.
5. If you need a batch-quality reference, compare against `paper_mse`.

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