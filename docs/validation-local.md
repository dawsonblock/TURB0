# Local validation on Apple Silicon

Public CI in this repository checks packaging and static validation. It does not certify the MLX runtime path, because generic hosted runners are not Apple Silicon and do not provide a usable `mlx` environment.

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
| **MLX structural** (`make test-mlx`, `make test-structural`, `make test-path-proof`) | KVCompressor, pipeline, calibration, streaming attention, path proof | Apple Silicon only |
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

# Full runtime certification (requires TQ_TEST_LLAMA_MODEL and TQ_TEST_GEMMA_MODEL)
make certify-apple-runtime
```

## Structural test files

`make test-structural` runs an explicit file list — no model weights required:

| File | What it proves |
|---|---|
| `test_path_not_dense_fallback.py` | TQ path is active; dense fallback is not silently taken |
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

For manual model smoke tests, run dense generation first, then the TurboQuant upgrade path on the same prompt and compare stability, memory use, and throughput.
