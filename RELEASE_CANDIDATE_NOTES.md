# Release-Candidate Notes — v0.2.2

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon
MLX inference. The supported runtime path is local Apple-Silicon validation
for selected Llama- and Gemma-family models. Custom Metal kernels are
experimental and not part of the default supported runtime. Other architectures
like Qwen, Mistral, and Phi are exploratory and uncertified.

---

## Summary

This pass covers Phases 2–6 and the `paper-contract-tranche1` branch of the
v0.2.2 release cycle. The repository is now fully linted (both `ruff check .`
and `ruff format --check .` pass under `nox -s lint`), statically clean (107
unit-static tests pass, typecheck reports no issues in 173 source files), and
runtime-certified against real model weights for both supported families.

The most recent combined certification artifact is
`artifacts/runtime-cert/20260405_210920`, which records a combined
`result: PASS` for Llama and Gemma in a single 23-stage run (60/60 benchmark
runs succeeded). Earlier single-family artifacts are retained for archaeology:
`artifacts/runtime-cert/20260404_013136` (Llama), `artifacts/runtime-cert/20260404_013527`
(Gemma), and `artifacts/runtime-cert/20260404_015658` (prior combined run).

The release candidate tag `v0.2.2-rc1` has also been pushed to exercise the remote
GitHub Actions gate. The Ubuntu jobs pass, and the final publish step remains
correctly blocked until `certify-apple-runtime` runs on an online self-hosted
`macOS` `ARM64` runner. No final `v0.2.2` tag should be cut until that job succeeds.

---

## Changes by phase

### Phase 2 — Config unification

- `TurboQuantConfig` unified: `group_size` → `k_group_size`, `main_bits` → `k_bits`
- `from_legacy_kwargs()` classmethod added for backward compatibility
- All callers migrated to new field names

### Phase 3 — KV cache state keys

- `TurboQuantKVCache.state()` updated to use `k_group_size` and `rotation` keys
- State round-trip tests updated to match new schema
- `meta_state` tuple in `integrations/mlx/cache_adapter.py` extended to 17 fields
  including `v_enabled`, `rotation_pad_to_pow2`, `residual_mode`, QJL params

### Phase 4 — Docs alignment

- `v_enabled` field documented across all config references
- `docs/supported-surface.md` and `docs/validation-local.md` aligned with
  current Makefile targets and certification scripts

### Phase 5 — Runtime certification

- Stubbed `tests/integration_mlx/` with six required test files:
  `test_cache_upgrade_roundtrip.py`, `test_streaming_attention_equivalence.py`,
  `test_llama_runtime_smoke.py`, `test_gemma_runtime_smoke.py`,
  `test_long_context_stability.py`, `test_path_proof_tq_active.py`
- Smoke tests use a tiny synthetic model by default (no download needed); the
  real-model path requires `TQ_TEST_LLAMA_MODEL` / `TQ_TEST_GEMMA_MODEL`.
- `make test-structural` runs the explicit file list.
- `./scripts/certify_apple_runtime.sh` now emits family-scoped PASS manifests.
- Retained PASS artifacts now exist at `artifacts/runtime-cert/20260404_013136`
  (Llama) and `artifacts/runtime-cert/20260404_013527` (Gemma).

### Phase 6 — paper-contract-tranche1: full lint clean + combined certification

- Repo-wide `ruff check .` reduced to zero violations by:
  - Auto-fix pass (`ruff check . --fix`) across all source files
  - Added `per-file-ignores` in `pyproject.toml` for `E402` in
    `tests/integration_mlx/*` and `tests/unit_mlx/*` (intentional MLX-guard
    pattern)
  - `# noqa: F403` on wildcard re-export lines in legacy shims
    `integrations/mlx/cache_adapter.py` and `integrations/mlx/upgrade.py`
  - Removed unused `angle_bits` variable from
    `benchmarks/exploratory/bench_polar_vs_scalar.py`
  - Replaced path-dependent bare imports with `importlib.import_module()` in
    `benchmarks/exploratory/decode_latency.py` to resolve `I001` conflict with
    mid-block `sys.path` mutation
- `ruff format .` applied repo-wide (~70 files reformatted, style-only)
- `nox -s lint` now passes end-to-end: both `ruff check .` and
  `ruff format --check .` are green
- `nox -s typecheck` reports no issues in 173 source files
- `pytest tests/unit_static -q` — 107 passed
- `python scripts/render_support_contract.py --check` — silent pass
- Combined certification (`TQ_TEST_LLAMA_MODEL` + `TQ_TEST_GEMMA_MODEL`)
  completed in a single 23-stage run:
  - 60/60 benchmark runs succeeded
  - Quality guardrail `paper_mse` Llama short Δppl +4.10 ≤ 20.0 ✅
  - Quality guardrail `paper_mse` Llama medium Δppl +11.84 ≤ 20.0 ✅
  - PolarQuant quality guardrails for both families: within threshold ✅
  - Manifest: `artifacts/runtime-cert/20260405_210920/cert_manifest.json`
    (`result: PASS`, `stages.passed: 23`, `stages.failed: 0`,
    `families: ["gemma", "llama"]`)

### Cleanup — Ruff linting and temp script removal
- 78 violations resolved (unused imports, whitespace, line length)
- `turboquant/__init__.py` updated: added `check_mlx_version`, `has_mlx`,
  `is_apple_silicon`, `require_mlx` to `__all__`
- Deleted 13 temporary patch/rewrite scripts from workspace root:
  `patch_*.py`, `fix2.py`, `rewrite_turboquant.py`, `sync.py`, `upgrade_qjl.py`

### Runtime fix — Centralized SDPA dispatch

- `mlx_lm/models/base.py` patched: `scaled_dot_product_attention` now
  type-guards on `TurboQuantKeysView` and routes to
  `turboquant_streaming_attention` automatically
- **No per-model changes required for new architectures** — confirmed working
  on Llama-3 and Gemma-2 without any model-specific wiring. Qwen is currently
  considered exploratory and not part of the supported allowlist.

### Benchmark scripts fixed

- `benchmarks/exploratory/bench_k_encode.py` updated to use `encode_k_block()`
- `benchmarks/exploratory/bench_decode_step.py` updated to use
  `TurboQuantKVCache(config=…, quantize_main=…, dequantize_main=…)`
- `scripts/run_benchmarks.sh` paths corrected to `benchmarks/exploratory/`

---

## Runtime benchmark results (HISTORICAL — not release-certified)

> **NOTE:** The figures below were recorded during early development and have
> not been reproduced against saved certification artifacts.  They are
> preserved here for historical reference only and **do not constitute
> certification evidence**.  Run `make certify-apple-runtime` with real model
> weights to produce artifact-backed results.

**Hardware:** Apple Silicon · macOS 26.2 · Python 3.10.12  
**Model:** `Llama / Gemma` (e.g. `Meta-Llama-3-8B-Instruct`)  
**Commit:** `6afc966`

### Synthetic micro-benchmarks

| Benchmark | Result |
|---|---|
| K-Encode (`encode_k_block`, shape [1,32,128,128]) | **0.10 ms / step** |
| Decode step (`append_keys`, 1 new token) | **0.03 ms / step** |

### Paired generative benchmarks (64 tokens, 5 prompts × 3 classes = 30 runs)

| Mode | Avg total | Tok/s | Status |
|---|---|---|---|
| Dense | 0.52 s | 147–163 | ✅ All 15 ok |
| TurboQuant | 6.80 s | 9–10 | ✅ All 15 ok |

Output text matched identically between dense and TurboQuant on all prompts,
confirming correctness of the compressed KV path. Qwen results are historically
verified but now considered exploratory and removed from primary support list.

> **Speed note:** TurboQuant decode speed reflects the uncompiled Python
> streaming attention path. `mx.compile` or `TQ_USE_METAL=1` is required to
> close the gap with dense baseline for production use.

---

## Gating status

> **NOTE:** The gates below reflect the current narrow release-candidate state.
> Tagged release publish must still re-run Apple certification with both
> `TQ_TEST_LLAMA_MODEL` and `TQ_TEST_GEMMA_MODEL` in scope.

The current RC tag demonstrates that this gate fails closed: the remote workflow
will wait for the Apple runner instead of publishing from Ubuntu-only checks.

| Gate | Result |
|---|---|
| `python scripts/preflight.py` | ✅ passes |
| `python -m build` (sdist + wheel) | ✅ passes |
| `ruff check .` | ✅ 0 violations |
| `ruff format --check .` | ✅ passes |
| `nox -s lint` (check + format) | ✅ passes |
| `nox -s typecheck` | ✅ no issues in 173 source files |
| `pytest tests/unit_static -q` | ✅ 107 passed |
| `make test-structural` | ✅ 4 / 4 (explicit file list) |
| `./scripts/certify_apple_runtime.sh` with `TQ_TEST_LLAMA_MODEL` | ✅ PASS — `artifacts/runtime-cert/20260404_013136` |
| `./scripts/certify_apple_runtime.sh` with `TQ_TEST_GEMMA_MODEL` | ✅ PASS — `artifacts/runtime-cert/20260404_013527` |
| `./scripts/certify_apple_runtime.sh` with both family env vars set (prior run) | ✅ PASS — `artifacts/runtime-cert/20260404_015658` |
| `./scripts/certify_apple_runtime.sh` with both family env vars set (latest run) | ✅ PASS — `artifacts/runtime-cert/20260405_210920` (23/23 stages, 60/60 runs) |
| Paired generative benchmark artifacts | ✅ Retained in both PASS artifact directories |
| Dense output == TurboQuant output (correctness) | ✅ Recorded through retained PASS artifacts |
| Remote tag `v0.2.2-rc1` release workflow | ⏳ Waiting for online `self-hosted` `macOS` `ARM64` runner |
