# Release checklist

TurboQuant is a research-grade KV-cache compression package for a narrow Apple-Silicon MLX runtime slice.
This checklist is the minimum bar for calling a tagged snapshot technically credible.

## Static gate

- `python scripts/preflight.py` passes
- `python tools/audit_vendored_surface.py` passes
- `python -m compileall turboquant mlx_lm tests` passes
- `python -m build` produces both sdist and wheel
- `python -m pytest tests/unit_static -q` passes
- the generated support docs still match `turboquant/contract.json`

## Apple Silicon structural gate

- `python -m pytest tests/integration_mlx -k "not llama and not gemma" -q` passes
- `make test-path-proof` passes

## Apple Silicon runtime gate

Use `./scripts/certify_apple_runtime.sh` as the authoritative release runtime gate.

Release publication remains blocked unless the same tagged workflow runs the Apple-Silicon certification job,
produces an evidence directory under `artifacts/runtime-cert/<timestamp>/`, and validates a real
`cert_manifest.json` with `result: PASS`.

Required release evidence includes:

- `contract.json`
- `cert_manifest.json`
- `preflight.json`
- `junit_cache_roundtrip.xml`
- `junit_attention_equiv.xml`
- `junit_llama_smoke.xml`
- `junit_gemma_smoke.xml`
- `junit_long_context.xml`
- `aggregate_runs.csv`
- `certification_summary.json`

Source archives do not embed those generated artifact directories. Release claims must point at a workflow artifact,
release evidence bundle, or pinned manifest digest.

Final tagged publish is stricter than a family-scoped local run:

- the manifest must record `result: PASS`
- the manifest must include both `llama` and `gemma` in `certification_scope.families`
- the self-hosted `macOS` `ARM64` runner pool must have been online for that same workflow

## Regression gate

- non-power-of-two rotation remains orthogonal
- legacy compatibility helpers still warn instead of silently widening the supported surface
- state save or restore still rejects config drift
- `block_tokens` remains compatibility-only rather than becoming an undocumented live control
- the `generate()` convenience defaults remain aligned to the paper-facing QJL path

## Documentation gate

- no primary doc publishes empirical benchmark numbers without provenance
- README and the generated support docs tell the same paper-facing story
- Llama and Gemma evidence depth is described honestly
- direct adapter construction is marked internal or compatibility-only rather than peer public API
- validation commands in docs still match the Makefile and certification scripts