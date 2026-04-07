# Theory And Traceability

This document is the theory-facing surface for TurboQuant's bounded research
story. It does not widen the supported runtime contract. The supported product
boundary still lives in `turboquant/contract.json`, and release-facing proof
still requires an addressable workflow artifact or manifest digest rather than
this source tree alone.

The goal here is narrower: make the paper-facing path explicit, checkable, and
auditable.

## Scope

TurboQuant currently has two distinct stories:

- Product contract: a narrow Apple-Silicon MLX runtime path for allowlisted
  Llama and Gemma families.
- Research traceability: a paper-facing scalar-plus-residual path and an
  experimental PolarQuant path that can be measured and compared without
  broadening support.

This document covers the second story. It should answer:

- which modules implement the paper-facing ideas
- which tests and benchmarks currently measure them
- which claims are implemented, which are only measured empirically, and which
  are still open

## The Paper-Facing Two-Stage Path

The repo's paper-facing production-style path is `paper_prod_qjl`.

At a high level it does four things:

1. Precondition keys with a deterministic orthogonal rotation.
2. Quantize the rotated keys with a scalar quantizer.
3. Form a residual in rotated space: `residual = rotated_key - scalar_reconstruction`.
4. Estimate residual score contributions with a 1-bit QJL sign sketch plus the
   stored residual norm.

The scalar-only comparison path is `paper_mse`. It uses the same scalar stage
but stops after step 2 and therefore has no residual correction stage.

The relationship between the two paths is important:

- `paper_mse` measures what the scalar stage alone does to distortion.
- `paper_prod_qjl` measures what changes when the residual QJL stage is added
  back into inner-product scoring.

That relationship is implemented today and directly measurable. What is not yet
promoted to a repo-level invariant is the stronger claim that the combined
two-stage path is already demonstrated here to be an unbiased inner-product
estimator. The repo now has a named bias-analysis lane for that question, but it
should stay an empirical result until repeated evidence supports a stronger
claim.

## Theorem-To-Code Map

| Claim or mechanism | Primary module | Validation surface | Current repo status |
| :--- | :--- | :--- | :--- |
| Orthogonal preconditioning of keys and queries | `turboquant/core/rotation.py` | `tests/unit_mlx/test_rotation_roundtrip.py` | Implemented and directly checked by roundtrip and orthogonality-style tests |
| Scalar quantization main stage for paper-facing presets | `turboquant/core/quantizer.py` | `tests/unit_mlx/test_mse_pipeline_roundtrip.py` | Implemented and measured through bounded reconstruction error checks |
| Paper-facing preset wiring for scalar-only vs scalar-plus-QJL modes | `turboquant/config.py` | `tests/unit_static/test_config_contract.py`, `tests/unit_static/test_algorithm_modes.py` | Implemented and contract-checked |
| Residual QJL sign sketch, norm storage, and residual score estimation | `turboquant/core/qjl.py` | `tests/unit_mlx/test_qjl.py`, `tests/unit_mlx/test_qjl_unbiasedness.py` | Implemented and empirically measured; current repo story is sign correlation and scale response, not proven unbiasedness |
| Two-stage paper attention scoring (`main_scores + residual_scores`) | `turboquant/core/residual_codec.py`, `turboquant/runtime/attention.py` | `tests/unit_mlx/test_attention_score_block_qjl.py`, `tests/unit_mlx/test_qjl_inner_product_bias_analysis.py` | Implemented and now measured as a named bias-analysis lane |
| Experimental PolarQuant path | `turboquant/core/polar_quant.py` | `tests/unit_mlx/test_polar_pipeline_roundtrip.py`, `tests/integration_mlx/test_polar_long_context_runtime.py`, `tests/integration_mlx/test_polar_gemma_runtime.py` | Implemented and supported as a non-paper-facing experimental branch |
| KV-cache runtime and benchmark evidence | `benchmarks/runtime_cert/run_dense_vs_tq.py`, `benchmarks/runtime_cert/run_quality_eval.py` | `scripts/certify_apple_runtime.sh`, runtime-cert artifacts, dated benchmark snapshots under `docs/history/` | Implemented with provenance-backed artifacts |
| Vector-search evidence | not yet implemented | planned research lane only | Open; no current repo evidence |

## Measured Versus Proven

The repo should stay explicit about what it knows today.

| Topic | Repo status today |
| :--- | :--- |
| Rotation preserves structure well enough for the supported path | Implemented and directly checked |
| Scalar stage has bounded empirical reconstruction error in current tests | Measured empirically |
| QJL sign sketch captures useful residual score signal | Measured empirically |
| The combined two-stage path is already proven unbiased in this repo | Open; do not claim this yet |
| PolarQuant is part of the paper-facing preset story | No; it is a supported non-paper-facing experimental branch |
| The repo reproduces the full breadth of the original TurboQuant framing, including vector search | No; current product and evidence remain KV-cache-first |

## Named Validation Lanes

These are the current repo surfaces that check the theory-facing pieces.

- `tests/unit_mlx/test_mse_pipeline_roundtrip.py`: bounded error checks for the
  scalar main stage.
- `tests/unit_mlx/test_qjl.py`: QJL packing, encode, decode, and shape checks.
- `tests/unit_mlx/test_qjl_unbiasedness.py`: current isolated QJL sketch checks;
  sign correlation and scale response, with an explicit note that the sketch is
  not treated as a proven unbiased estimator here.
- `tests/unit_mlx/test_qjl_inner_product_bias_analysis.py`: deterministic bias
  snapshot for the actual two-stage `paper_prod_qjl` attention estimator versus
  the scalar-only `paper_mse` baseline.
- `benchmarks/runtime_cert/run_dense_vs_tq.py`: paired dense-vs-TurboQuant
  runtime sweeps used for dated benchmark snapshots and release-proof bundles.

## Open Claims

These are the important theory-facing questions that remain open in this repo.

1. Whether the combined two-stage residual path is unbiased enough to be stated
   as a repo-level invariant rather than only a measured empirical result.
2. Whether the current paper-facing path generalizes beyond the narrow
   allowlisted runtime slice without widening support prematurely.
3. Whether vector-search outcomes can be added as research validation without
   confusing them with the supported Apple runtime contract.

Until those are answered with retained evidence, the strongest honest claim is
that the repo contains a bounded, measurable TurboQuant-style implementation
with explicit traceability from theory-facing ideas to code and tests.