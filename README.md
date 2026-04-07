<div align="center">

# TurboQuantX1

**Contract-driven KV-cache compression for Apple-Silicon MLX LLMs**

[![Python](https://img.shields.io/badge/python-3.9--3.11-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-%3E%3D0.30.0%20%3C1.0.0-orange)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://apple.com/mac)
[![Version](https://img.shields.io/badge/version-0.2.2-green)](RELEASE_CANDIDATE_NOTES.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

*Paper-facing presets: `paper_prod`, `paper_mse` · Supported non-paper-facing branch: `polarquant_exp`*

<p>
    <a href="#current-slice">Current Slice</a> ·
    <a href="#runtime-contract">Runtime Contract</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#validation-and-certification">Validation</a> ·
    <a href="#documentation-map">Docs</a>
</p>

</div>

TurboQuantX1 is a research-stage KV-cache compression library for transformer
inference on Apple Silicon via
[mlx-lm](https://github.com/ml-explore/mlx-lm). The repository contains
exploratory code, compatibility paths, and a vendored `mlx_lm` tree, but the
formal support claim is intentionally narrower than the codebase footprint.

> **Accuracy rule**
>
> The machine-readable source of truth is `turboquant/contract.json`. The
> generated contract docs summarize it. Generated `artifacts/runtime-cert/` bundles are workflow outputs, and built wheels and source distributions do not ship those generated directories. Neither a source
> archive nor a built distribution proves a current PASS without an addressable
> certification artifact, release evidence bundle, or pinned manifest digest.

## Current Slice

| Topic | Current contract |
|---|---|
| Platform | `darwin-arm64` on Apple Silicon |
| Python / MLX | Python `3.9`-`3.11`, MLX `>= 0.30.0` and `< 1.0.0` |
| Canonical runtime path | `upgrade_cache_list(...)` inside the `mlx_lm` decode flow |
| Allowlisted model families | `llama`, `gemma` |
| Paper-facing presets | `paper_prod`, `paper_mse` |
| Supported non-paper-facing branch | `polarquant_exp` |
| Compatibility-only surfaces | `legacy_topk`, direct adapter construction, deprecated cache-conversion helpers |
| Out of scope | blanket vendored-model support, non-Apple runtimes, production deployment claims |

| What this repo is | What this repo is not |
|---|---|
| A narrow, contract-validated runtime slice | A general-purpose LLM runtime |
| A research-stage Apple-Silicon MLX integration | Blanket support for every vendored `mlx_lm` model family |
| A repo with explicit release-evidence rules | Proof of a current PASS without published evidence |

## Runtime Contract

Contract summary: TurboQuant supports one canonical runtime path for
allowlisted Llama and Gemma models via `upgrade_cache_list(...)` inside the
`mlx_lm` decode flow. Direct adapter construction and deprecated cache
conversion helpers remain secondary surfaces that bypass the support gate.

The validated promotion path is:

`generate_step(...)` -> `maybe_turboquant_k_cache(...)` -> `upgrade_cache_list(...)` -> `TurboQuantKCache.update_and_fetch(...)` -> `TurboQuantKeysView` -> `scaled_dot_product_attention(...)` -> `turboquant_streaming_attention(...)`

Important boundaries:

- `upgrade_cache_list(...)` is the canonical support-gated entry point.
- `TurboQuantKCache(...)`, `KVCache._to_turboquant()`, and `KVCache.to_turboquant()` remain compatibility or eval surfaces, not peer public runtime APIs.
- The canonical decode path returns runtime-upgrade events, but it does not automatically persist `events.jsonl`.
- Evidence depth is intentionally asymmetric today: Llama coverage is stronger; Gemma remains narrower overall because the conservative `paper_mse` batch quality guardrail is still Llama-scoped even though PolarQuant runtime and quality stages now run on both families.

The supported public cache-state persistence format is
`TurboQuantKVCache.state()` with `schema_version == 4`.

Generated summaries of the contract live in
[docs/product_contract.md](docs/product_contract.md),
[docs/support_matrix.md](docs/support_matrix.md), and
[docs/supported-surface.md](docs/supported-surface.md).

## Presets And Branches

| Surface | Contract status | Residual | Use it when |
|---|---|---|---|
| `paper_prod` / `paper_prod_qjl` | paper-facing | `qjl` | You want the primary production-style research path |
| `paper_mse` | paper-facing | `none` | You want the conservative reference path and batch-quality guardrail |
| `polarquant_exp` | supported non-paper-facing | `none` | You want the PolarQuant runtime path with family-scoped certification |
| `balanced`, `max_quality`, `legacy_topk` | compatibility-only | `topk` | You are loading historical configs or comparing legacy behavior |

Notes:

- `high_compression` remains a legacy alias for the `paper_prod_qjl` family.
- `balanced`, `max_quality`, and `legacy_topk` are legacy top-k compatibility-only surfaces.
- `polarquant_exp` is part of the formal supported product contract, but it is not part of the paper-facing preset story.
- Exact preset math and deviations are generated in [docs/support_matrix.md](docs/support_matrix.md).

## Quick Start

Apple Silicon is required for runtime inference. Non-Apple platforms are for
static checks, linting, and contract-oriented verification only.

```bash
git clone https://github.com/dawsonblock/TURB0.git
cd TURB0
pip install -e '.[apple]'
```

For static or development work without the Apple runtime extras:

```bash
pip install -e '.[dev]'
```

Canonical cache upgrade after prefill:

```python
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from turboquant.config import TurboQuantConfig
from turboquant.integrations.mlx.upgrade import upgrade_cache_list

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
cache = make_prompt_cache(model)

# ... run prefill here ...

cfg = TurboQuantConfig.from_preset("paper_prod")
events = upgrade_cache_list(
    cache,
    k_start=64,
    config=cfg,
    model_family="llama",
)
```

The same runtime path also supports PolarQuant for allowlisted families:

```python
cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
events = upgrade_cache_list(cache, k_start=64, config=cfg, model_family="gemma")
```

If you use the higher-level `mlx_lm.generate.generate(...)` wrapper, it still
delegates into the same upgrade machinery.

## Model Support

| Family | Status | Evidence depth | Current coverage |
|---|---|---|---|
| Llama | allowlisted | stronger | real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, `paper_mse` batch quality guardrail, long-context stability, dense-vs-TurboQuant benchmark sweeps |
| Gemma | allowlisted | narrower | real-model smoke, PolarQuant runtime smoke, PolarQuant quality guardrail, dense-vs-TurboQuant benchmark sweeps; narrower overall because the conservative `paper_mse` gate is still Llama-only |

Families present in the vendored `mlx_lm` tree are not automatically supported.
Allowlist membership is a contract decision, not a side effect of vendored code.

## Validation And Certification

If you only validated this repo from a non-Apple or no-MLX environment, the
strongest honest claim is that the package build/install structure and
non-runtime validation lanes work there. That is useful, but it is not a
runtime go/no-go for the Apple-Silicon MLX path.

This repository layout and its static checks do not, by themselves, prove a
current Apple runtime PASS for any release. Only a published certification
artifact or pinned manifest digest from the tagged Apple-arm64 workflow does
that.

Local validation entry points:

```bash
make test-static
make test-mlx
make test-structural
make test-path-proof
make test-smoke-llama
make test-smoke-gemma
make test-long-context
```

On Apple Silicon, the smoke targets above default to `TinyModel` when the
real-model environment variables are unset.

Real-model smoke targets:

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"

make test-smoke-llama
make test-smoke-gemma
make test-long-context
```

Full Apple-Silicon certification bundle:

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"
bash scripts/certify_apple_runtime.sh
```

Important release-evidence rules:

- Generated `artifacts/runtime-cert/` bundles are workflow outputs; built wheels and source distributions do not ship those generated directories.
- `scripts/certify_apple_runtime.sh` snapshots `contract.json` into the artifact directory so the evidence bundle carries both the run result and the exact contract it satisfied.
- Primary docs should not publish timeless benchmark tables without addressable provenance.
- Benchmark claims require an artifact or manifest plus commit, model ids, MLX version, hardware, script, and invocation arguments.

For the full evidence model, read
[docs/runtime-certification.md](docs/runtime-certification.md),
[docs/validation-local.md](docs/validation-local.md), and
[docs/benchmark_methodology.md](docs/benchmark_methodology.md).

## Documentation Map

| Document | Purpose |
|---|---|
| [docs/theory.md](docs/theory.md) | Theory-facing map of paper claims, implementation anchors, and current evidence limits |
| [docs/product_contract.md](docs/product_contract.md) | Generated top-level product boundary |
| [docs/supported-surface.md](docs/supported-surface.md) | Generated canonical vs secondary surface definition |
| [docs/support_matrix.md](docs/support_matrix.md) | Generated family and preset matrix |
| [docs/runtime-certification.md](docs/runtime-certification.md) | Certification scope, stages, and evidence contract |
| [docs/architecture.md](docs/architecture.md) | Runtime path and component map |
| [docs/integration.md](docs/integration.md) | Model-family wiring and PolarQuant integration details |
| [docs/evaluation.md](docs/evaluation.md) | Exploratory quality-evaluation guidance |
| [docs/validation-local.md](docs/validation-local.md) | Local validation walkthrough |
| [docs/benchmark_methodology.md](docs/benchmark_methodology.md) | Benchmark publication and provenance rules |
| [VENDORED_MLX_LM.md](VENDORED_MLX_LM.md) | Vendored mlx-lm boundary and patch notes |

## Project Layout

```text
TurboQuantX1/
├── turboquant/
│   ├── config.py
│   ├── core/
│   ├── runtime/
│   ├── integrations/mlx/
│   ├── eval/
│   └── kernels/
├── mlx_lm/
├── benchmarks/
│   ├── exploratory/
│   └── runtime_cert/
├── tests/
├── scripts/
└── docs/
```

## Contributing

1. Run `make test-static` on any platform.
2. On Apple Silicon, run `make test-mlx` and `make test-structural` before widening claims.
3. If you change runtime-contract or evidence wording, update `turboquant/contract.json` and regenerate the derived docs.
4. If you add a model family or preset to the supported story, extend the certification surface before updating the README.

## License

[MIT](LICENSE)
