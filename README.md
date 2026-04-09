<div align="center">

# TurboQuant

### Research-grade KV-cache compression for Apple Silicon MLX LLMs

[![Static CI](https://github.com/dawsonblock/TURB0/actions/workflows/static-ci.yml/badge.svg)](https://github.com/dawsonblock/TURB0/actions/workflows/static-ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-%3E%3D0.30.0%2C%3C1.0.0-FF6600?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://apple.com/mac)
[![Version](https://img.shields.io/badge/version-0.2.2-22C55E)](RELEASE_CANDIDATE_NOTES.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Compress transformer KV-caches at decode time on Apple Silicon — with a narrow, contract-validated runtime path.**

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Presets](#presets) · [Model Support](#model-support) · [Validation](#validation--certification) · [Docs](#documentation)

</div>

---

TurboQuant is a **research-stage** KV-cache compression library that plugs into [`mlx-lm`](https://github.com/ml-explore/mlx-lm) inference on Apple Silicon. It patches the upstream decode loop at import time and routes allowlisted model families through a compressed attention path — no per-model fork required.

> **Scope note** — The machine-readable source of truth is [`turboquant/contract.json`](turboquant/contract.json). This repository does not prove a current Apple runtime PASS without a published certification artifact or pinned manifest digest from the tagged `apple-runtime-cert` workflow.

---

## Quick Start

> **Apple Silicon is required for runtime inference.** All other platforms support static checks, linting, and contract validation only.

```bash
git clone https://github.com/dawsonblock/TURB0.git
cd TURB0

# Apple Silicon — full runtime
pip install -e '.[apple]'

# Any platform — static / dev work only
pip install -e '.[dev]'
```

### Upgrade a cache after prefill

```python
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from turboquant.config import TurboQuantConfig
from turboquant.integrations.mlx.upgrade import upgrade_cache_list

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
cache = make_prompt_cache(model)

# ... run your prefill here ...

cfg = TurboQuantConfig.from_preset("paper_prod")
events = upgrade_cache_list(
    cache,
    k_start=64,
    config=cfg,
    model_family="llama",
)
```

The same path works for PolarQuant:

```python
cfg = TurboQuantConfig.polarquant_exp(rotation="random_orthogonal")
events = upgrade_cache_list(cache, k_start=64, config=cfg, model_family="gemma")
```

The higher-level `mlx_lm.generate.generate(...)` wrapper delegates into the same machinery automatically once the patch layer is active.

---

## How It Works

TurboQuant patches three upstream symbols at import time — no vendored fork:

```
mlx_lm.models.cache.make_prompt_cache   (patched)
mlx_lm.generate.generate_step           (patched)
         │
         ▼
upgrade_cache_list(...)          ← canonical support-gated entry point
         │
         ▼
TurboQuantKCache.update_and_fetch(...)
         │
         ▼
TurboQuantKeysView
         │
         ▼
mlx_lm.models.base.scaled_dot_product_attention   (patched)
         │
         ▼
turboquant_streaming_attention(...)
```

The attention fast path scores flat K-history slices from runtime-packed tensors and decodes V in chunks with an online softmax (log-sum-exp streaming reduction), avoiding a full dense V concatenation at every decode step.

**Contract summary:**

- `upgrade_cache_list(...)` is the canonical, support-gated entry point.
- `TurboQuantKCache(...)` is internal/eval-only — it bypasses the model-family allowlist.
- `KVCache.to_turboquant()` is mentioned here as documentation shorthand for the cache-adapter upgrade path; it is **not** currently a shipped/runtime-available `KVCache` method. Use `upgrade_cache_list(...)` for supported upgrades.
- The decode path returns `events` but does not automatically persist `events.jsonl`.
- Cache state is persisted as `TurboQuantKVCache.state()` at `schema_version == 4`.

---

## Presets

| Preset | Classification | K bits | V bits | Residual | Use when |
|---|---|:---:|:---:|---|---|
| `paper_prod` / `paper_prod_qjl` | paper-facing | 3 | 4 | QJL (1-bit) | Primary two-stage research path |
| `paper_mse` | paper-facing | 3 | 4 | none | Conservative scalar-only reference |
| `polarquant_exp` | supported, non-paper-facing | 3 | 4 | none | PolarQuant with family-scoped certification |
| `legacy_topk`, `balanced`, `max_quality` | compatibility-only | — | — | top-k | Loading historical configs only |

- `paper_prod` is a stable alias for `paper_prod_qjl`.
- `high_compression` is a legacy alias for the `paper_prod_qjl` family.
- `polarquant_exp` is a formally supported contract surface but is outside the paper-facing preset story.
- Generated preset math lives in [docs/support_matrix.md](docs/support_matrix.md).

---

## Model Support

| Family | Status | Evidence depth | Coverage |
|---|---|---|---|
| **Llama** | ✅ allowlisted | stronger | real-model smoke · PolarQuant runtime & quality · `paper_mse` batch guardrail · long-context stability · dense-vs-TQ benchmark sweeps |
| **Gemma** | ✅ allowlisted | narrower | real-model smoke · PolarQuant runtime & quality · dense-vs-TQ benchmark sweeps |

> Families reachable through the patch layer are **not** automatically supported. Allowlist membership is a contract decision, not a side effect of patch reachability. See [`turboquant/runtime/support.py`](turboquant/runtime/support.py) and [`turboquant/contract.json`](turboquant/contract.json).

Gemma coverage is intentionally narrower: the conservative `paper_mse` batch quality guardrail is still Llama-scoped.

---

## Validation & Certification

### Static checks (any platform)

```bash
make test-static        # static unit test suite, no MLX required
make compile            # bytecode check across all source + test modules
```

### Apple Silicon runtime checks

```bash
make test-structural    # path-proof, cache roundtrip, streaming attention — no model weights
make test-path-proof    # verify TQ path is exercised, not dense fallback
make test-smoke-llama   # Llama smoke (TinyModel by default)
make test-smoke-gemma   # Gemma smoke (TinyModel by default)
make test-long-context  # long-context stability (TinyModel by default)
make test-mlx           # full MLX suite
```

### Real-model smoke (Apple Silicon)

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"

make test-smoke-llama
make test-smoke-gemma
make test-long-context
```

### Full certification bundle

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"
bash scripts/certify_apple_runtime.sh
```

> **Evidence rule** — `artifacts/runtime-cert/` bundles are workflow outputs; built wheels and source distributions do not ship that directory. Static CI passing on Linux is not a runtime go/no-go — source and built snapshots do not, by themselves, prove a current Apple runtime PASS. Only a published certification artifact or pinned manifest digest from a tagged `apple-runtime-cert` workflow run proves a current PASS for both allowlisted families.

---

## Project Layout

```text
TURB0/
├── turboquant/
│   ├── config.py                  # TurboQuantConfig — the runtime config API
│   ├── contract.json              # machine-readable support contract
│   ├── patch.py                   # upstream mlx_lm patch bootstrap
│   ├── core/                      # rotation, quantizer, QJL, PolarQuant
│   ├── runtime/                   # attention fast path, support gate
│   ├── integrations/mlx/          # upgrade_cache_list, cache adapter
│   ├── eval/                      # logit comparison helpers
│   └── kernels/                   # experimental Metal kernel stubs
├── benchmarks/
│   ├── exploratory/               # micro-benchmarks and ablations
│   └── runtime_cert/              # certification benchmark scripts
├── tests/
│   ├── unit_static/               # contract / structural tests (no MLX)
│   ├── unit_mlx/                  # unit tests requiring MLX
│   └── integration_mlx/          # full-path integration tests
├── scripts/                       # certify_apple_runtime.sh, validate_local.sh
├── tools/                         # dist verification, surface audit
└── docs/                          # generated and hand-written documentation
```

---

## Documentation

| Document | Purpose |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Runtime path and component map |
| [docs/theory.md](docs/theory.md) | Paper-claim traceability and current evidence limits |
| [docs/product_contract.md](docs/product_contract.md) | Generated top-level product boundary |
| [docs/support_matrix.md](docs/support_matrix.md) | Generated family and preset matrix |
| [docs/supported-surface.md](docs/supported-surface.md) | Generated canonical vs secondary surface definitions |
| [docs/preset_modes.md](docs/preset_modes.md) | Generated preset taxonomy |
| [docs/runtime-certification.md](docs/runtime-certification.md) | Certification scope, stages, and evidence contract |
| [docs/validation-local.md](docs/validation-local.md) | Local validation walkthrough |
| [docs/benchmark_methodology.md](docs/benchmark_methodology.md) | Benchmark publication and provenance rules |
| [docs/benchmark_index.md](docs/benchmark_index.md) | Generated index of benchmark surfaces and lane boundaries |
| [docs/family_evidence_matrix.md](docs/family_evidence_matrix.md) | Release-gated vs research-only evidence split |
| [docs/integration.md](docs/integration.md) | Model-family wiring and PolarQuant integration details |
| [docs/evaluation.md](docs/evaluation.md) | Exploratory quality-evaluation guidance |
| [docs/bit_budget_sweep.md](docs/bit_budget_sweep.md) | Research-only bit-budget sweep |
| [docs/kv_paper_eval.md](docs/kv_paper_eval.md) | Unified KV report command (fast-check vs heavy-offline tiers) |
| [docs/vector_search.md](docs/vector_search.md) | Research-only vector-search benchmark lane |
| [docs/vendored-upstream-boundary.md](docs/vendored-upstream-boundary.md) | Upstream `mlx_lm` patch boundary |

---

## Contributing

1. Run `make test-static` on any platform before opening a PR.
2. On Apple Silicon, run `make test-mlx` and `make test-structural` before widening any runtime claims.
3. If you change runtime-contract or evidence wording, update `turboquant/contract.json` and regenerate the derived docs.
4. If you change the preset registry or classifications, update `turboquant/config.py` and `turboquant/contract.json` together, then regenerate the derived docs.
5. If you add a model family or preset to the supported story, extend the certification surface **before** updating the README.

---

## License

[MIT](LICENSE)
