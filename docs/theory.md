# Theory And Traceability

This document is the theory-facing surface for TurboQuant's bounded research
story. It does not widen the supported runtime contract. The supported product
boundary still lives in `turboquant/contract.json`, and release-facing proof
still requires an addressable workflow artifact or manifest digest rather than
this source tree alone.

The goal here is narrower: make the paper-facing path explicit, checkable, and
auditable.

## Lane Separation

TurboQuant currently has two distinct lanes:

- Product lane: a narrow Apple-Silicon MLX runtime path for allowlisted Llama
  and Gemma families, release-gated by the machine-readable contract and
  runtime-cert evidence.
- Research lane: theory traceability, paper-facing presets, bias analysis,
  ablations, and future vector-search work used to measure alignment with the
  paper without widening support.

This document belongs to the research lane. When a code path is shared by both
lanes, the theory-facing claim still stays research-scoped unless the product
contract explicitly says otherwise.

## Status Labels

| Label | Meaning |
| :--- | :--- |
| `implemented` | The code path exists and is directly exercised by a named repo surface. |
| `empirical` | The repo measures the behavior with tests or benchmarks, but it is still an observed result rather than a theorem-level proof. |
| `partial` | Some implementation and evidence exist, but the surface is narrower than the full paper framing or still missing important evidence. |
| `not yet shown` | The repo does not currently provide the implementation or retained evidence needed for the claim. |

## Claims Status Table

| Paper idea | Repo module(s) | Benchmark or test coverage | Status | Scope | Current repo claim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random rotation stage | `turboquant/core/rotation.py`, `turboquant/core/pipeline.py` | `tests/unit_mlx/test_rotation_roundtrip.py` | `implemented` | `research-only` | The repo has an explicit rotation stage with direct roundtrip and orthogonality-style checks, but it does not promote a theorem claim from implementation alone. |
| Scalar MSE stage | `turboquant/core/quantizer.py`, `turboquant/core/pipeline.py` | `tests/unit_mlx/test_mse_pipeline_roundtrip.py` | `empirical` | `research-only` | The scalar stage is implemented and bounded by empirical reconstruction checks on the current validation slice. |
| Residual 1-bit QJL stage | `turboquant/core/qjl.py`, `turboquant/core/residual_codec.py` | `tests/unit_mlx/test_qjl.py`, `tests/unit_mlx/test_qjl_unbiasedness.py` | `empirical` | `research-only` | The sign sketch and residual norm bookkeeping are implemented and measurably affect score estimates, but the repo does not yet claim unbiasedness. |
| Scalar-only paper path (`paper_mse`) | `turboquant/config.py`, `turboquant/core/quantizer.py` | `tests/unit_static/test_config_contract.py`, `tests/unit_static/test_algorithm_modes.py`, `tests/unit_mlx/test_mse_pipeline_roundtrip.py` | `empirical` | `research-only` | `paper_mse` is an explicit scalar-only comparison path and a valid baseline for bias and quality experiments. |
| Full two-stage paper path (`paper_prod_qjl`) | `turboquant/config.py`, `turboquant/core/qjl.py`, `turboquant/core/residual_codec.py`, `turboquant/runtime/attention.py` | `tests/unit_static/test_config_contract.py`, `tests/unit_mlx/test_attention_score_block_qjl.py`, `tests/unit_mlx/test_qjl_inner_product_bias_analysis.py` | `partial` | `research-only` | The path is implemented and benchmarkable, but the stronger theorem-style claim about unbiased inner-product restoration remains open here. |
| Current inner-product-bias evidence | `benchmarks/runtime_cert/run_inner_product_bias_eval.py`, `tests/unit_mlx/test_qjl_inner_product_bias_analysis.py` | `inner_product_bias_summary.json`, `inner_product_bias_metrics.csv`, `inner_product_bias_summary.md`, dated benchmark snapshots under `docs/history/`, MLX bias tests | `empirical` | `research-only` | The repo retains signed-error, absolute-error, and variance measurements for scalar-only versus two-stage scoring, but treats them as research diagnostics rather than release truth. |
| Current KV-cache evidence | `benchmarks/runtime_cert/run_dense_vs_tq.py`, `benchmarks/runtime_cert/run_quality_eval.py`, `scripts/certify_apple_runtime.sh` | runtime-cert artifacts, `tests/integration_mlx/test_long_context_stability.py`, dated benchmark snapshots under `docs/history/` | `partial` | `product` | The repo has addressable Apple-MLX evidence for allowlisted families, but that evidence is narrow, family-scoped, and does not imply universal speed or quality wins. |
| Missing vector-search evidence | not yet implemented | none | `not yet shown` | `research-only` | The repo does not yet have a retained vector-search benchmark lane, recall results, or memory-and-latency evidence for that part of the paper framing. |

## Code-To-Claim Mapping

| Code surface | Claim exercised here | Primary presets or lane | Evidence surface | Current limit |
| :--- | :--- | :--- | :--- | :--- |
| `turboquant/config.py` | Preset wiring and algorithm taxonomy for scalar-only, two-stage, and non-paper experimental paths | `paper_mse`, `paper_prod_qjl`, `polarquant_exp` | `tests/unit_static/test_config_contract.py`, `tests/unit_static/test_algorithm_modes.py` | It defines the reachable surfaces; it does not itself prove distortion, bias, or quality properties. |
| `turboquant/core/rotation.py` | Orthogonal preconditioning stage | paper-facing presets and `polarquant_exp` | `tests/unit_mlx/test_rotation_roundtrip.py` | Rotation is implemented and checked, but downstream paper claims remain empirical. |
| `turboquant/core/quantizer.py` | Scalar quantization main stage | `paper_mse`, `paper_prod_qjl` | `tests/unit_mlx/test_mse_pipeline_roundtrip.py` | Reconstruction bounds are measured on the repo's current fixtures, not proved for all settings. |
| `turboquant/core/qjl.py` | Residual 1-bit sign sketch and norm scaling | `paper_prod_qjl` | `tests/unit_mlx/test_qjl.py`, `tests/unit_mlx/test_qjl_unbiasedness.py` | The repo keeps the stronger unbiasedness claim explicitly open. |
| `turboquant/core/residual_codec.py`, `turboquant/runtime/attention.py` | Two-stage score composition: scalar main scores plus residual score estimate | `paper_prod_qjl` | `tests/unit_mlx/test_attention_score_block_qjl.py`, `tests/unit_mlx/test_qjl_inner_product_bias_analysis.py`, `benchmarks/runtime_cert/run_inner_product_bias_eval.py` | Current evidence is measurement, not theorem-level proof. |
| `turboquant/core/polar_quant.py` | PolarQuant-only non-paper branch | `polarquant_exp` | `tests/unit_mlx/test_polar_pipeline_roundtrip.py`, `tests/integration_mlx/test_polar_long_context_runtime.py`, `tests/integration_mlx/test_polar_gemma_runtime.py` | This branch is supported as experimental and explicitly outside the paper-facing preset story. |
| `benchmarks/runtime_cert/run_dense_vs_tq.py` | KV-cache memory and throughput tradeoff on the allowlisted Apple-MLX path | `product` lane | runtime-cert artifacts, `docs/history/BENCHMARK_SNAPSHOT_*.md` | The retained story is memory relief on a narrow hardware and family slice; throughput can regress materially. |
| `benchmarks/runtime_cert/run_quality_eval.py` | Family-scoped quality guardrails used during certification and research comparison | `product` lane, family-scoped | `quality_eval_*_summary.json` artifacts | These guardrails catch regressions; they do not prove generalized quality improvement. |
| `benchmarks/runtime_cert/run_inner_product_bias_eval.py` | Retained bias snapshot for scalar-only versus two-stage scoring | `research` lane | `inner_product_bias_summary.json`, `inner_product_bias_metrics.csv`, `inner_product_bias_summary.md`, dated benchmark snapshots | The current retained workload is synthetic; real KV-derived bias evidence is not yet retained here. |
| `benchmarks/vector_search/` | Vector-search recall, memory, and indexing claims | planned research lane only | none yet | Not implemented in this repo today. |

## Still Not Proven Here

These are the important theory-facing questions that remain open in this repo.

1. Whether the combined two-stage residual path is strong enough to promote a
   repo-level unbiased-inner-product claim rather than only an empirical bias
   measurement lane.
2. Whether the current paper-facing path generalizes beyond the narrow
   allowlisted Apple-MLX runtime slice without widening support prematurely.
3. Whether vector-search outcomes can be added as research validation without
   confusing them with the supported product lane.
4. Whether the current evidence depth for Gemma should ever be described as
   equal to Llama; today it should not.
5. Whether TurboQuant should ever be described here as a throughput win on the
   current uncompiled Apple-MLX path; current retained evidence does not
   justify that claim.

Until those questions are answered with retained evidence, the strongest honest
statement is that the repo contains a bounded, measurable TurboQuant-style
implementation with explicit traceability from paper-facing ideas to code,
tests, and research artifacts, while the supported product contract remains
narrower than the research story.
