<div align="center">

# TurboQuantX1

**Experimental KV-cache compression for Apple Silicon MLX LLMs**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.30.0%2B-orange)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://apple.com/mac)
[![Version](https://img.shields.io/badge/version-0.2.2-green)](RELEASE_CANDIDATE_NOTES.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

*3-bit keys · 4-bit values · Hadamard-family rotation · QJL production path · legacy top-k compatibility only*

</div>

---

## Table of Contents

- [What is TurboQuantX1?](#what-is-turboquantx1)
- [Runtime Contract](#runtime-contract)
- [Paper-Facing Presets](#paper-facing-presets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [State And Compatibility](#state-and-compatibility)
- [Evaluation](#evaluation)
- [Testing And Certification](#testing-and-certification)
- [Benchmark Provenance](#benchmark-provenance)
- [Model Support](#model-support)
- [Project Layout](#project-layout)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## What is TurboQuantX1?

TurboQuantX1 is a research-stage KV-cache compression library for transformer models running on
Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm). It is not a general-purpose
LLM runtime, not production-ready, and not broadly validated across the vendored `mlx_lm` tree.
Its current support claim is narrow: local Apple-Silicon MLX validation for allowlisted
**Llama-family** and **Gemma-family** models on one canonical upgrade path.

The paper-facing runtime story is:

| Layer | Current story |
|---|---|
| Key quantization | Group scalar quantization with Hadamard-family rotation |
| Production-style residual | `paper_prod` / `paper_prod_qjl` with a 1-bit QJL residual |
| Batch-quality reference | `paper_mse` with no residual |
| Legacy compatibility | top-k residual presets remain available, but only as compatibility surfaces |
| Attention | streaming decode over compressed keys via `TurboQuantKeysView` |

> **Status:** Narrow release candidate. Source archives document the certification workflow,
> but they do not prove a current PASS without an addressable workflow artifact, release
> evidence bundle, or pinned manifest digest. The release workflow is designed to produce
> Apple-arm64 certification artifacts for `llama` and `gemma`, but the evidence depth is not
> symmetrical today: Llama coverage is stronger, Gemma coverage is narrower because the current
> batch quality guardrail remains Llama-scoped. Custom Metal kernels remain experimental
> (`TQ_USE_METAL=1`). Full boundary: [docs/supported-surface.md](docs/supported-surface.md).

Contract summary: TurboQuant supports one canonical runtime path for allowlisted Llama and
Gemma models via `upgrade_cache_list(...)` inside the `mlx_lm` decode flow. Direct
`TurboQuantKCache(...)` construction, `KVCache._to_turboquant()`, and
`KVCache.to_turboquant()` remain secondary compatibility or eval surfaces that bypass the
support gate. Runtime upgrade decisions and persisted certification logs are separate layers,
and the canonical decode path does not automatically persist `events.jsonl`. `block_tokens`
is retained for compatibility but does not currently affect the attention dispatch path.

---

## Runtime Contract

The supported runtime promotion path is:

`generate_step(...)` -> `maybe_turboquant_k_cache(...)` -> `upgrade_cache_list(...)` -> `TurboQuantKCache.update_and_fetch(...)` -> `TurboQuantKeysView` -> `scaled_dot_product_attention(...)` -> `turboquant_streaming_attention(...)`

Important boundaries:

- `upgrade_cache_list(...)` is the canonical support-gated entry point.
- `TurboQuantKCache(...)` is an internal adapter, not a peer public runtime API.
- `KVCache._to_turboquant()` and `KVCache.to_turboquant()` are deprecated compatibility helpers.
- `block_tokens` is compatibility-only and is not a live hot-path control.
- The canonical decode path returns lightweight runtime events and does not automatically persist `events.jsonl`.

The machine-readable source of truth for this boundary is
`turboquant/contract.json`. The generated docs in
[docs/product_contract.md](docs/product_contract.md),
[docs/support_matrix.md](docs/support_matrix.md), and
[docs/supported-surface.md](docs/supported-surface.md) are derived from that file.

---

## Paper-Facing Presets

Paper-facing presets are `paper_prod` and `paper_mse`. Legacy top-k presets remain available for
compatibility and experimentation, but they are not the primary story presented by the repo.

| Preset | Canonical algorithm | Residual | Average KV bpc at `d=128` | Notes |
|---|---|---|:---:|---|
| `paper_prod` | `paper_prod_qjl` | `qjl` | 3.562 | Paper-facing production-style path |
| `paper_mse` | `paper_mse` | `none` | 3.75 | Paper-facing MSE-only reference path |
| `high_compression` | `paper_prod_qjl` | `qjl` | 3.562 | Legacy alias for the paper-facing production-style preset |
| `balanced` | `legacy_topk` | `topk` | compatibility-only | Legacy top-k preset |
| `max_quality` | `legacy_topk` | `topk` | compatibility-only | Legacy top-k preset |

Exact deviations from the paper-facing story:

- Non-power-of-two head dimensions use a deterministic orthogonal fallback instead of an exact Hadamard matrix.
- Legacy aliases, `residual_topk`, and `block_tokens` remain for compatibility, but they are not part of the paper-facing preset contract.
- The vendored `mlx_lm` tree is much wider than the supported boundary; only allowlisted families are eligible for the canonical path.
- `legacy_topk` and `polarquant_exp` remain implementation branches, not the repo's primary paper-facing claim.

---

## Installation

Apple Silicon Mac (M1, M2, M3, or M4) is required for inference. Non-Apple platforms support
static analysis, linting, type-checking, and static contract tests only.

```bash
git clone https://github.com/dawsonblock/TURB0.git
cd TURB0
pip install -e '.[apple]'
```

Non-Apple platform:

```bash
pip install -e '.[dev]'
```

Full development environment on Apple Silicon:

```bash
pip install uv nox
uv pip install -e '.[apple,dev]'
```

---

## Quick Start

### Option 1: `generate()` convenience wrapper over the canonical path

The `mlx_lm.generate.generate(...)` surface still uses legacy kwarg names. Setting
`turboquant_residual_topk=0` on that surface selects the paper-facing QJL path.

```python
from mlx_lm import generate, load

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

response = generate(
    model,
    tokenizer,
    prompt="Explain KV-cache compression in one paragraph.",
    max_tokens=256,
    turboquant_k_start=64,
    turboquant_k_bits=3,
    turboquant_group_size=64,
    turboquant_rotation="hadamard",
    turboquant_residual_topk=0,
    turboquant_v_bits=4,
    turboquant_v_group_size=64,
    turboquant_model_family="llama",
)
```

This route still delegates to the same support-gated runtime path:
`generate_step(...)` -> `maybe_turboquant_k_cache(...)` -> `upgrade_cache_list(...)`.

### Option 2: canonical manual cache upgrade after prefill

```python
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from turboquant.config import TurboQuantConfig
from turboquant.integrations.mlx.upgrade import upgrade_cache_list

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
cache = make_prompt_cache(model)

# ... run prefill here ...

cfg = TurboQuantConfig.from_preset("paper_prod")
events = upgrade_cache_list(cache, k_start=64, config=cfg, model_family="llama")

for evt in events:
    print(
        f"Layer {evt.layer_index}: {evt.old_type} -> {evt.new_type} "
        f"(upgraded={evt.upgraded}, offset={evt.offset_at_upgrade})"
    )
```

### Option 3: explicit paper-facing presets

```python
from turboquant.config import TurboQuantConfig

cfg_prod = TurboQuantConfig.from_preset("paper_prod")
cfg_mse = TurboQuantConfig.from_preset("paper_mse")
```

Use `paper_prod` for the production-style QJL path and `paper_mse` for the batch-quality
reference path.

---

## Configuration Reference

All runtime behavior is controlled by `TurboQuantConfig`.

```python
from turboquant.config import TurboQuantConfig

cfg = TurboQuantConfig(
    algorithm="paper_prod_qjl",
    k_bits=3,
    k_group_size=64,
    v_bits=4,
    v_group_size=64,
    v_enabled=True,
    rotation="hadamard",
    residual_mode="qjl",
    qjl_proj_dim=64,
    return_mode="view",
)
cfg.validate()
```

Key fields:

| Field | Default | Notes |
|---|---|---|
| `algorithm` | `paper_prod_qjl` | Canonical families: `paper_mse`, `paper_prod_qjl`, `legacy_topk`, `polarquant_exp` |
| `k_bits` / `v_bits` | `3` / `4` | Main paper-facing bit-widths |
| `rotation` | `hadamard` | `hadamard`, `random_orthogonal`, or `identity` |
| `residual_mode` | `qjl` | `qjl`, `none`, or `topk` |
| `qjl_proj_dim` | `64` | Production-style paper-facing residual projection size |
| `residual_topk` | `0` | Compatibility-only for legacy top-k mode |
| `block_tokens` | `256` | Compatibility-only knob; not read by the hot path |
| `return_mode` | `view` | Canonical runtime path always upgrades into the streaming view mode |

Legacy aliases still normalize to the canonical algorithm families:

- `paper_prod` -> `paper_prod_qjl`
- `turboquant_prod` -> `paper_prod_qjl`
- `turboquant_mse` -> `paper_mse`

---

## State And Compatibility

Runtime cache state is versioned and validated. Current state payloads carry `schema_version == 4`.

```python
state = cache.state()
assert state["schema_version"] == 4

from turboquant.runtime.state import validate_state

validate_state(state)
```

Compatibility notes:

- `validate_state()` still accepts older flat v1, v2, and v3 payloads for migration checks.
- `block_tokens` is retained so old configs and historical scripts still load, but it is not a live tuning lever.
- `KVCache.to_turboquant()` and `_to_turboquant()` remain available only as deprecated compatibility helpers.

---

## Evaluation

```python
from turboquant.config import TurboQuantConfig
from turboquant.eval import drift_report, memory_report, perplexity_report

cfg = TurboQuantConfig.from_preset("paper_prod")

ppl = perplexity_report(model, input_ids, turboquant_config=cfg, model_family="llama")
drift = drift_report(model, input_ids, turboquant_config=cfg, model_family="llama")
mem = memory_report(model, input_ids, turboquant_config=cfg, model_family="llama")
```

Local default checks used by `run_quality_eval.py`:

| Check | Default | Scope |
|---|---|---|
| Perplexity delta | `Δppl <= 0.5` | Local script default |
| Mean KL divergence | `mean_kl <= 0.1` | Local script default |

These are local exploratory defaults, not certification guarantees and not release gates.
For the current runtime-cert story, see [docs/runtime-certification.md](docs/runtime-certification.md)
and [docs/validation-local.md](docs/validation-local.md).

---

## Testing And Certification

```bash
# Platform-agnostic static suite
make test-static

# Apple-Silicon MLX tests
make test-mlx

# Structural integration tests
make test-structural
make test-path-proof

# Smoke and stability targets
make test-smoke-llama
make test-smoke-gemma
make test-long-context
```

Smoke targets default to TinyModel on Apple Silicon. To switch the same targets to real weights:

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"

make test-smoke-llama
make test-smoke-gemma
make test-long-context
```

Exploratory real-model batch-quality probes are opt-in:

```bash
export TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY=1
python -m pytest tests/integration_mlx/test_dense_vs_paper_mse_275bpc.py -v
python -m pytest tests/integration_mlx/test_dense_vs_paper_mse_375bpc.py -v
```

Full runtime certification:

```bash
make certify-structural
make certify-apple-runtime
```

The certification script snapshots `contract.json` into the generated evidence directory so that
release artifacts carry both the machine-readable contract and the run results together.

---

## Benchmark Provenance

Primary docs in this repo do not publish empirical benchmark tables unless each published number maps to an
addressable evidence bundle or manifest digest plus exact commit, model ids, MLX version, hardware, script,
and invocation arguments.

That means this README intentionally links to commands and provenance rules instead of preserving historical
latency or memory tables as if they were current facts.

Benchmark entry points:

```bash
python benchmarks/exploratory/bench_memory_footprint.py
python benchmarks/exploratory/bench_dense_vs_turboquant.py
python benchmarks/exploratory/bench_decode_streaming.py
python benchmarks/exploratory/bench_decode_step.py
python benchmarks/exploratory/bench_k_encode.py
```

Reproduction and publication rules live in [docs/benchmark_methodology.md](docs/benchmark_methodology.md).

---

## Model Support

Current supported slice:

| Family | Canonical status | Evidence depth | Notes |
|---|---|---|---|
| Llama | allowlisted | stronger | real-model smoke, batch quality guardrail, long-context stability, dense-vs-TurboQuant sweeps |
| Gemma | allowlisted | narrower | real-model smoke and dense-vs-TurboQuant sweeps; current batch quality guardrail remains Llama-scoped |

Unsupported families such as Qwen, Mistral, and Phi may exist in the vendored `mlx_lm` tree, but they are not
eligible for the canonical `upgrade_cache_list(...)` path today. Routing through vendored attention code is not the
same thing as allowlist membership.

---

## Project Layout

```text
TurboQuantX1/
├── turboquant/
│   ├── config.py
│   ├── core/
│   ├── runtime/
│   ├── integrations/mlx/
│   │   ├── _cache_adapter.py
│   │   ├── cache_adapter.py
│   │   └── upgrade.py
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

---

## Documentation

| Document | Purpose |
|---|---|
| [docs/product_contract.md](docs/product_contract.md) | Generated top-level product boundary |
| [docs/supported-surface.md](docs/supported-surface.md) | Generated canonical vs secondary surface definition |
| [docs/support_matrix.md](docs/support_matrix.md) | Generated family and preset matrix |
| [docs/runtime-certification.md](docs/runtime-certification.md) | Evidence, scope, and certification workflow |
| [docs/benchmark_methodology.md](docs/benchmark_methodology.md) | Benchmark publication and provenance rules |
| [docs/architecture.md](docs/architecture.md) | Runtime path and component map |
| [docs/integration.md](docs/integration.md) | Model-family wiring guide |
| [docs/evaluation.md](docs/evaluation.md) | Exploratory quality evaluation guide |
| [docs/validation-local.md](docs/validation-local.md) | Local validation walkthrough |
| [VENDORED_MLX_LM.md](VENDORED_MLX_LM.md) | Vendored mlx-lm boundary and patch notes |

---

## Contributing

1. Run `make test-static` on any platform.
2. On Apple Silicon, run `make test-mlx` and `make test-structural`.
3. If you touch runtime-contract or evidence wording, update the machine-readable contract and regenerate the derived docs.
4. If you add a model family, wire its attention path, extend runtime certification, update the support contract, and then update the docs generated from it.

---

## License

[MIT](LICENSE)