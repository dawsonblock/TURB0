# Contract Status

## Current status

The paper contract is now unified around a single authority file: `turboquant/contract.json`.

The checked-in generated docs, the runtime support gate, the certification script, the Apple certification workflow, and the tagged release workflow all consume or validate against that same contract.

## Proven in the current tree

- Canonical runtime path: `turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)`
- Supported platform slice: Apple Silicon `darwin-arm64`
- Supported runtime slice: Python 3.9 through 3.11, MLX `>= 0.30.0` and `< 1.0.0`
- Supported families: `llama` and `gemma` only
- Paper-facing presets: `paper_mse` and `paper_prod` / `paper_prod_qjl`
- Release workflow checks both `cert_manifest.json` and the retained `contract.json` snapshot, and for tagged releases it also requires both allowlisted families in `certification_scope.families`

Retained evidence already present in this workspace:

- `artifacts/runtime-cert/20260404_013136` — `PASS` for `llama`
- `artifacts/runtime-cert/20260404_013527` — `PASS` for `gemma`
- `artifacts/runtime-cert/20260404_015658` — combined `PASS` with `certification_scope.families=["gemma", "llama"]`

## Compatibility-only or experimental

- `turboquant.integrations.mlx._cache_adapter.TurboQuantKCache`
- `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache`
- `mlx_lm.models.cache.KVCache._to_turboquant()`
- `mlx_lm.models.cache.KVCache.to_turboquant()`
- `turboquant.eval.compare._collect_logits_compressed()`
- Legacy compatibility branches such as `legacy_topk` and `polarquant_exp`
- Exploratory real-model `paper_mse` quality tests under `tests/integration_mlx/test_dense_vs_paper_mse_275bpc.py` and `tests/integration_mlx/test_dense_vs_paper_mse_375bpc.py`

Those surfaces remain available for compatibility or investigation, but they are outside the supported contract.

## Validation executed for this cleanup

Validation was run in `.venv-cert311` on Apple Silicon with Python 3.11.13 and MLX 0.31.1.

- `python scripts/render_support_contract.py --check` — passed
- `python scripts/preflight.py` — passed
- `python -m pytest tests/unit_static -q --tb=short` — `92 passed`
- `python -m pytest tests/integration_mlx -q -k "not llama and not gemma" --tb=short` — `15 passed, 4 deselected`
- `python -m build` — passed after installing the missing `build` frontend into `.venv-cert311`

The long-running real-model certification workflow was not rerun as part of this doc-and-contract cleanup because retained PASS evidence already exists under `artifacts/runtime-cert/` and the current changes did not alter the underlying certification stages.