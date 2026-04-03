# Local validation on Apple Silicon

Public CI in this repository checks packaging and static validation. It does not certify the MLX runtime path, because generic hosted runners are not Apple Silicon and do not provide a usable `mlx` environment.

If you do not run the Apple-Silicon path yourself, you do not have runtime evidence. You only have static evidence.

## The one real proof you can run today

```bash
make test-path-proof
```

This runs `tests/integration_mlx/test_path_not_dense_fallback.py` — the only
structural proof that the TurboQuant path is active and not silently falling back
to dense computation.  It requires no model weights, produces results in under
10 seconds on any Apple Silicon Mac, and is the narrowest meaningful check in
the repo.

## Two-track testing model

| Track | What it tests | Where it runs |
|---|---|---|
| **Static** (`make test-static`) | Import smoke, version consistency, source-checkout preflight, schema-level checks | Any platform |
| **MLX structural** (`make test-mlx`, `make test-structural`, `make test-path-proof`) | KVCompressor, pipeline, calibration, streaming attention; offset tracking; cache independence | Apple Silicon only |
| **Model smoke** (`make test-smoke-llama`, `make test-smoke-gemma`, `make test-long-context`) | End-to-end generation with TurboQuant active; NaN guards; dense-fallback detection | Apple Silicon + model env vars |
| **Runtime certification** (`make certify-apple-runtime`) | Artifact-producing release validation, smoke runs, benchmarks, and metric aggregation | Apple Silicon only |

## Quick start

```bash
# Static tests (safe everywhere)
make test-static

# Narrowest Apple Silicon proof (start here — no model weights needed)
make test-path-proof

# Full structural validation
make test-mlx
make test-structural

# Model smoke tests (skip cleanly if env vars absent)
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
make test-smoke-llama    # Llama end-to-end with TurboQuant active
make test-long-context   # long-context (>256 tokens) — NaN and fallback checks

# Full runtime certification (requires TQ_TEST_LLAMA_MODEL and TQ_TEST_GEMMA_MODEL)
make certify-apple-runtime
```

## Structural test files

`make test-structural` runs an explicit file list — no model weights required:

| File | What it proves |
|---|---|
| `test_path_not_dense_fallback.py` | TQ path is active; dense fallback not silently taken; offset accumulates monotonically across sequential appends; two cache instances are independent |
| `test_cache_upgrade_roundtrip.py` | Cache state round-trips through upgrade without data loss |
| `test_streaming_attention_equivalence.py` | Streaming attention output matches reference on synthetic tensors |

## Validation scripts

`./scripts/validate_apple_silicon.sh`

- local developer validation
- creates a fresh virtualenv, installs the package in editable mode with Apple extras
- Lanes: preflight → MLX unit tests → path-proof gate → cache + attention tests → optional model smoke tests

`./scripts/certify_apple_runtime.sh`

- release certification
- writes timestamped artifacts under `artifacts/runtime-cert/`
- fails closed: stages where all tests are `@skip` are counted as UNIMPLEMENTED, not PASSED
- requires `TQ_TEST_LLAMA_MODEL` and `TQ_TEST_GEMMA_MODEL` env vars for full certification

## Legacy integration tests

`tests/integration/` is a dead path — it no longer exists. The canonical release-certification paths are `tests/unit_static/`, `tests/unit_mlx/`, and `tests/integration_mlx/`.

## Manual smoke testing

Use `make test-smoke-llama` (Llama) or `make test-smoke-gemma` (Gemma) for scripted smoke runs.
These assert: token output > 0, at least one `TurboQuantKCache` layer is upgraded (dense fallback
detected if none), and no exception during generation. Use `make test-long-context` to confirm
NaN-free logprobs on contexts that exceed the default block size (256 tokens).
