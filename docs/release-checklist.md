# Release checklist

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation for selected Llama-family and Gemma-family models. Custom Metal kernels are experimental and not part of the default supported runtime.

This is the minimum bar for calling a tagged snapshot technically credible. It is a release gate, not a wish list.

## Static gate

Run from a fresh checkout.

- `python scripts/preflight.py` passes
- `python tools/audit_vendored_surface.py` passes
- `python -m compileall turboquant mlx_lm tests` passes
- `python -m build` produces both sdist and wheel
- `python -m pytest tests/unit_static -q` passes
- `README.md`, `docs/supported-surface.md`, and `docs/validation-local.md` agree on the supported slice

## Apple Silicon structural gate

Run on an Apple Silicon Mac with MLX installed.

- `python -m pytest tests/unit_mlx/ -q` passes
- `python -m pytest tests/integration_mlx -k "not llama and not gemma" -q` passes
- `make test-path-proof` passes

## Apple Silicon runtime gate

Use the certification script as the authoritative release runtime gate.

Tagged release publish is blocked unless the same `release.yml` workflow runs a
self-hosted Apple-Silicon certification job, uploads the generated artifact directory,
and validates a real `cert_manifest.json` with `result: PASS`. Ubuntu static checks alone
must never publish a tag.

The first retained PASS artifact may be family-scoped while certification widens. The
manifest must record which families were in scope for that run, Llama should be proven first,
and every in-scope stage must pass.
Tagged release publish is stricter than local artifact generation: the release workflow must
run with both `TQ_TEST_LLAMA_MODEL` and `TQ_TEST_GEMMA_MODEL` set and validate that the
resulting manifest includes both `llama` and `gemma` in `certification_scope.families`.

If the self-hosted runner pool is offline, the tag must remain blocked in GitHub Actions.
Do not substitute prior local artifacts, prior workflow artifacts, or a manual judgment call
for the same-workflow Apple certification job.

Release-candidate tags may exercise the release workflow gates, but they must not execute the
`publish` job. Only the final stable tag should attempt PyPI publish and GitHub release creation.

That combined-family gate is now demonstrated by the retained local artifact
`artifacts/runtime-cert/20260404_015658/`.

- `./scripts/certify_apple_runtime.sh` passes
- `cert_manifest.json` exists in the retained certification artifact directory and records `result: PASS`
- At least one Llama-family smoke run succeeds when Llama is in scope and `TQ_TEST_LLAMA_MODEL` is set
- At least one Gemma-family smoke run succeeds when Gemma is in scope and `TQ_TEST_GEMMA_MODEL` is set
- Dense vs TurboQuant artifact outputs are saved under `artifacts/runtime-cert/<timestamp>/`
- `preflight.json` and all JUnit outputs are present in the certification artifact directory
- A runner with labels `self-hosted`, `macOS`, and `ARM64` is online before the final release tag is pushed

## Regression gate

- Non-power-of-two rotation remains orthogonal
- `residual_topk` survives the legacy adapter path
- state save/restore rejects config drift
- deprecated legacy knobs still warn instead of silently changing runtime behavior
- `python scripts/preflight.py` continues to work from a plain checkout

## Documentation gate

- No benchmark claim is labeled production unless it is backed by release data
- No CI badge implies MLX runtime certification on generic runners
- Supported models are named explicitly
- Validation commands in docs match the actual Makefile and Nox sessions
