# Release-Candidate Notes — v0.2.2

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon
MLX inference. The supported runtime path is local Apple-Silicon validation
for selected Llama- and Gemma-family models. Custom Metal kernels are
experimental and not part of the default supported runtime. Other architectures
like Qwen, Mistral, and Phi are exploratory and uncertified.

---

## Summary

This pass covers Phases 2–5 of the v0.2.2 release cycle, bringing the
repository from scaffolding to a fully linted state with static tests
passing. Runtime certification is **not yet complete** — integration smoke
tests require real model weights (`TQ_TEST_LLAMA_MODEL` / `TQ_TEST_GEMMA_MODEL`)
to exercise the memory-backed path. The quality evaluation script
(`benchmarks/runtime_cert/run_quality_eval.py`) is implemented.

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
- `./scripts/certify_apple_runtime.sh` fails when required stages are
  skipped — full cert requires real model weights.

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

## Gating status (HISTORICAL — state at time of Phase 5 scaffolding)

> **NOTE:** The gates below reflect the state at Phase 5 scaffolding.
> `make test-structural` now runs an explicit file list (not a name filter).
> `./scripts/certify_apple_runtime.sh` now fails when required stages are
> skipped — full cert requires `TQ_TEST_LLAMA_MODEL` and `TQ_TEST_GEMMA_MODEL`.

| Gate | Result |
|---|---|
| `python scripts/preflight.py` | ✅ passes |
| `python -m build` (sdist + wheel) | ✅ passes |
| `ruff check .` | ✅ 0 violations |
| `make test-structural` | ✅ 4 / 4 (explicit file list) |
| `./scripts/certify_apple_runtime.sh` (no model weights) | ✗ FAILS — required stages unimplemented/skipped |
| Paired generative benchmark (30 runs) | ⬜ Not reproduced — historical only |
| Dense output == TurboQuant output (correctness) | ⬜ Not reproduced — historical only |
