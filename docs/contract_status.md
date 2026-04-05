# Contract Status

## Current authority

The supported TurboQuant boundary is unified around a single machine-readable
authority file: `turboquant/contract.json`.

The checked-in generated docs, the runtime support gate, the packaging
verifier, the certification scripts, the Apple certification workflow, and the
tagged release workflow all consume or validate against that same contract.

## Supported public runtime surface

- Canonical runtime path: `turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)`
- Package-root MLX surface remains lazy and fail-closed; callers without MLX on
  Apple Silicon get an `ImportError` instead of a silent fallback.
- Supported platform slice: Apple Silicon `darwin-arm64`
- Supported runtime slice: Python 3.9 through 3.11, MLX `>= 0.30.0` and `< 1.0.0`
- Supported families: `llama` and `gemma` only
- Paper-facing presets: `paper_mse` and `paper_prod` / `paper_prod_qjl`
- Supported non-paper-facing branch: `polarquant_exp`, exercised through the
  same `upgrade_cache_list(...)` path with `TurboQuantConfig.polarquant_exp(...)`
  and the same `llama` / `gemma` allowlist gate.
- Published package boundary: the wheel intentionally ships the bounded
  `turboquant` package together with the vendored `mlx_lm` tree,
  `turboquant/contract.json`, and `mlx_lm/py.typed`; `docs/*.md` remain
  source-distribution-only for review.
- Release workflow checks both `cert_manifest.json` and the retained
  `contract.json` snapshot, and for tagged releases it also requires both
  allowlisted families in `certification_scope.families`.

## Retained release evidence

- `artifacts/runtime-cert/20260404_201202` — full promoted-contract `PASS` on
  `darwin-arm64` with `23/23` stages passed and
  `certification_scope.families=["gemma", "llama"]`.
- That retained bundle includes the required release evidence for both
  paper-facing presets and the supported non-paper-facing `polarquant_exp`
  branch, including `junit_polar_llama_runtime.xml`,
  `junit_polar_gemma_runtime.xml`, `quality_eval_polar_short_summary.json`,
  `quality_eval_polar_medium_summary.json`,
  `quality_eval_polar_gemma_short_summary.json`, and
  `quality_eval_polar_gemma_medium_summary.json`.
- Earlier retained checkpoints remain useful for archaeology, including
  `artifacts/runtime-cert/20260404_013136` (`llama`),
  `artifacts/runtime-cert/20260404_013527` (`gemma`), and
  `artifacts/runtime-cert/20260404_015658` (earlier combined `PASS`).

## Compatibility-only or secondary surfaces

- `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache`
- `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache`
- `mlx_lm.models.cache.KVCache._to_turboquant()`
- `mlx_lm.models.cache.KVCache.to_turboquant()`
- `turboquant.eval.compare._collect_logits_compressed()`
- Legacy compatibility branches such as `legacy_topk`
- Exploratory real-model `paper_mse` quality tests under `tests/integration_mlx/test_dense_vs_paper_mse_275bpc.py` and `tests/integration_mlx/test_dense_vs_paper_mse_375bpc.py`

Those surfaces remain available for compatibility or investigation, but they
are not the canonical support-gated public runtime path and should not be
treated as release-facing entry points.

## Validation executed for this cleanup

Validation for this boundary-hardening pass was run in `.venv-cert311` on
2026-04-04 and stayed confined to the packaging/docs/static lane plus the
already-retained Apple runtime evidence.

- `python scripts/render_support_contract.py --check` — passed
- `python -m build` — passed
- `python tools/verify_dist_contents.py` — passed
- `python -m compileall turboquant mlx_lm tests` — passed
- `pytest tests/unit_static -q` — `99 passed`

The long-running real-model certification workflow does not need to be rerun
for packaging/docs-only changes so long as the retained `PASS` bundle under
`artifacts/runtime-cert/20260404_201202` remains the current evidence source
and the runtime contract itself is unchanged.
