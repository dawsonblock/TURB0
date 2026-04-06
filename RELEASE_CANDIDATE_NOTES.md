# Release-Candidate Notes — v0.2.2

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon
MLX inference. The supported runtime path is local Apple-Silicon validation
for selected Llama- and Gemma-family models. Custom Metal kernels are
experimental and not part of the default supported runtime. Other architectures
like Qwen, Mistral, and Phi are exploratory and uncertified.

---

## Summary

This pass covers Phases 2–6 and the `paper-contract-tranche1` branch of the
v0.2.2 release cycle. The repository is now structurally aligned around a
bounded Apple-Silicon MLX support contract: the package surface, generated
contract docs, workflow gates, and static governance all point at the same
narrow runtime slice.

That structural work cannot prove a current Apple runtime PASS
for this portable source snapshot. This archive does not bundle
`artifacts/runtime-cert/`, and built wheels or source distributions would not
ship that directory anyway. Final release publication therefore remains
blocked until the tagged Apple-arm64 workflow publishes an addressable
certification artifact or pinned manifest digest whose `cert_manifest.json`
reports `PASS` for both allowlisted families (`llama` and `gemma`).

The current RC workflow design still fails closed in the right place: generic
Ubuntu validation can succeed while the final publish step remains blocked
until the self-hosted `macOS` `ARM64` runner completes certification for the
same tagged revision.

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

### Phase 5 — Runtime certification contract

- Stubbed `tests/integration_mlx/` with six required test files:
  `test_cache_upgrade_roundtrip.py`, `test_streaming_attention_equivalence.py`,
  `test_llama_runtime_smoke.py`, `test_gemma_runtime_smoke.py`,
  `test_long_context_stability.py`, `test_path_proof_tq_active.py`
- Smoke tests use a tiny synthetic model by default (no download needed); the
  real-model path requires `TQ_TEST_LLAMA_MODEL` / `TQ_TEST_GEMMA_MODEL`.
- `make test-structural` runs the explicit file list.
- `./scripts/certify_apple_runtime.sh` now emits family-scoped manifests and
  the Apple workflows validate both `cert_manifest.json` and the artifact's
  retained `contract.json` snapshot.
- Final tagged release remains blocked until the Apple-arm64 workflow produces
  and publishes a combined PASS artifact covering both supported families.

### Phase 6 — paper-contract-tranche1: full lint clean + release gate hardening

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
- `nox -s lint` was restored as the canonical Ruff-based generic lint lane
- Type annotations across the maintained support surface were tightened to
  keep the static contract legible and auditable
- `pytest tests/unit_static -q` and
  `python scripts/render_support_contract.py --check` remain the key
  structural proof points for this source snapshot
- The release workflows now require a two-family Apple-arm64 certification
  bundle before final publish; this archive documents that requirement but
  does not bundle the resulting evidence

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
> not been validated through the certification process.  They are
> preserved here for historical reference only and **do not constitute
> certification evidence**.  Run `make certify-apple-runtime` with real model
> weights to produce an addressable certification artifact or pinned manifest
> digest.

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

> **IMPORTANT:** The gates below distinguish structural completeness from
> evidential completeness. A portable source snapshot can be buildable and
> statically coherent without proving a current Apple runtime PASS.

| Gate | Meaning in this archive |
|---|---|
| `python -m build` (sdist + wheel) | Structural packaging proof, not runtime proof |
| `pytest tests/unit_static -q` | Structural support-contract proof, not runtime proof |
| `make test-structural` | Apple-Silicon structural gate without real model weights |
| Tagged Apple-arm64 certification artifact | Required release evidence; this archive does not include it |
| Published certification artifact or pinned manifest digest | Required for an unqualified release claim |
| Remote tag `v0.2.2-rc1` release workflow | Designed to fail closed until an online `self-hosted` `macOS` `ARM64` runner certifies the tag |
