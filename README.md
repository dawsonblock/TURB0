<div align="center">

# ⚡ TurboQuantX1

**Research-grade KV-cache compression for Apple Silicon MLX LLMs**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.30.0%2B-orange)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://apple.com/mac)
[![Version](https://img.shields.io/badge/version-0.2.2-green)](RELEASE_CANDIDATE_NOTES.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

*3-bit keys · 4-bit values · deterministic Hadamard rotation · top-k sparse residual · no NumPy in the hot path*

</div>

---

## Table of Contents

- [What is TurboQuantX1?](#what-is-turboquantx1)
- [Memory Compression](#memory-compression)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Presets](#presets)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Model Support Matrix](#model-support-matrix)
- [Hardware Requirements](#hardware-requirements)
- [Project Layout](#project-layout)
- [Component Status](#component-status)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Development Setup](#development-setup)
- [Contributing](#contributing)
- [License](#license)

---

## What is TurboQuantX1?

TurboQuantX1 is a research-grade KV-cache compression library for transformer models running on
Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm). It targets **memory reduction
first**, compressing the attention KV cache by **3.5×–4.4× versus dense fp16** using:

| Technique | What it does |
|---|---|
| **Group scalar quantization** | Per-group min/max scales at configurable bit-widths (2–8 bit) |
| **Hadamard-family rotation** | Orthogonal whitening that equalises per-dimension variance before quantization |
| **Top-k sparse residual** | Stores the k largest-magnitude errors the main quantizer misses, recovered at decode time |
| **Streaming attention** | Model-agnostic decode path that operates directly on compressed codes |

All compression runs within the MLX compute graph — no NumPy synchronization in the hot path.

> **Status:** Serious prototype targeting production quality on Apple Silicon. Supported runtime:
> local Apple Silicon validation for **Llama-family** and **Gemma-family** models. Custom Metal
> kernels are experimental (`TQ_USE_METAL=1`). Other architectures (Qwen, Mistral, Phi) are
> exploratory and uncertified. Full surface definition: [docs/supported-surface.md](docs/supported-surface.md).

---

## Memory Compression

Local illustrative measurements — not release-certified unless matched by saved artifacts in
`artifacts/runtime-cert/<timestamp>/`.

| Configuration | Tokens | Total MB | Bytes / Token | vs Dense |
|:---|:---:|:---:|:---:|:---:|
| Dense `float16` | 1 024 | 2.10 MB | 2 048 | 1.0× |
| k=4b, group=64 | 1 024 | 0.61 MB | 592 | **3.5×** |
| k=3b, group=64 | 1 024 | 0.57 MB | 560 | **3.7×** |
| k=2b, group=64 | 1 024 | 0.48 MB | 464 | **4.4×** |
| k=4b, group=32 | 1 024 | 0.67 MB | 656 | 3.1× |
| k=3b, group=32 | 1 024 | 0.64 MB | 624 | 3.3× |

**Detailed breakdown — 3-bit K, group=64, 1 024 tokens, 2 heads, head\_dim=128:**

```
k_packed           ~229 kB    3-bit packed uint32 codes
k_scales             ~8 kB    per-group fp16 scales
k_resid_values       ~8 kB    top-k fp16 residual values (k=2)
k_resid_indices      ~4 kB    top-k uint8 indices
v_packed           ~262 kB    4-bit packed uint32 codes
v_scales             ~8 kB    per-group fp16 scales
─────────────────────────────
total              ~519 kB    vs 2 048 kB dense  (~4× compression)
```

---

## How It Works

TurboQuantX1 wraps each KV-cache layer in a two-stage compressed representation. At decode time,
keys are reconstructed on-the-fly during attention.

```
                    K  path
┌──────────┐   ┌───────────────┐   ┌──────────────────────┐   ┌────────┐
│ raw keys │──▶│ FixedRotation │──▶│ GroupScalarQuantizer │──▶│ packed │
│[B,H,T,D] │   │ Hadamard / QR │   │ N-bit, per-group     │   │  codes │
└──────────┘   └───────────────┘   └──────────────────────┘   └────────┘
                                             │ residual error
                                             ▼
                                  ┌──────────────────────┐
                                  │  encode_topk_residual│
                                  │  top-k val + idx     │
                                  └──────────────────────┘

                    V  path
┌────────────┐   ┌──────────────────────┐   ┌────────┐
│ raw values │──▶│ GroupScalarQuantizer │──▶│ packed │
│ [B,H,T,D] │   │ M-bit, per-group     │   │  codes │
└────────────┘   └──────────────────────┘   └────────┘

Streaming decode (every attention call)
  packed_codes ──▶ dequant ──▶ + topk_residual ──▶ crop ──▶ [B,H,T,D]
  (queries are rotated with the same FixedRotation before the matmul)
```

### Core design choices

**Hadamard-family whitening** — exact dense Hadamard matrix for power-of-two head dims; a
deterministic Hadamard-derived orthogonal fallback otherwise. Satisfies `R.T @ R = I`. Cost is
O(d²) per token — not a fast butterfly transform.

**Top-k sparse residual** — after quantization the k=2 largest-magnitude per-group errors are
stored as fp16 values with uint8 indices. Added back at decode time before the attention matmul.

**Two-phase bit-packing** — pad to group boundary, then to uint32 word boundary. Handles
arbitrary bit-widths (including 3-bit) for any head dimension in vectorized MLX ops, no NumPy sync.

**Single execution path** — `.build()` pre-allocates buffers and binds operations once at init.
The config selects the encode/decode path at construction; zero runtime branches in the hot path.

**Centralized SDPA dispatch** — `mlx_lm/models/base.py`'s `scaled_dot_product_attention`
type-guards on `TurboQuantKeysView` and automatically routes to `turboquant_streaming_attention`.
No per-model changes required for new architectures.

**Versioned state schema** — all `state()` dicts carry `schema_version: 2`. `validate_state()`
enforces structural correctness on restore — raises rather than silently loading corrupt state.

---

## Architecture

```
turboquant/
│
├── config.py                  TurboQuantConfig — single source of truth
├── core/
│   ├── rotation.py            FixedRotation (Hadamard / QR / identity)
│   ├── quantizer.py           GroupScalarQuantizer — vectorized pack/unpack
│   ├── residual.py            encode_topk_residual / decode_topk_residual
│   └── pipeline.py            encode_k_block / decode_k_block — single path, no branches
├── runtime/
│   ├── layout.py              ensure_layout [B, H, T, D]
│   ├── kv_interface.py        TurboQuantKVCache + TurboQuantKeysView
│   ├── attention.py           turboquant_streaming_attention
│   ├── support.py             Model family allowlist + assert_supported_model_family()
│   └── state.py               STATE_SCHEMA_VERSION + validate_state()
├── integrations/mlx/
│   ├── cache_adapter.py       TurboQuantKCache  (mlx_lm protocol adapter)
│   └── upgrade.py             upgrade_cache_list()  —  canonical upgrade API
├── calibration/
│   └── fit_quantizer.py       calibrate()  —  offline scale fitting
├── eval/
│   ├── perplexity.py          perplexity_report()
│   ├── generation_drift.py    drift_report()
│   └── memory.py              memory_report()
└── kernels/                   Metal kernel dispatch (experimental, TQ_USE_METAL=1)

mlx_lm/                        Patched vendored mlx-lm v0.29.1
├── models/base.py             SDPA — TurboQuantKeysView auto-dispatch
├── models/llama.py            Llama wiring
├── models/gemma.py            Gemma wiring
└── generate.py                maybe_turboquant_k_cache + generate_step
```

Two integration surfaces:

| Surface | Entrypoint | When |
|---|---|---|
| **Cache upgrade** | `upgrade_cache_list(cache, ...)` | Once, after prefill completes |
| **Attention dispatch** | SDPA type-guard in `base.py` | Every decode step, automatically |

---

## Installation

> Apple Silicon Mac (M1/M2/M3/M4) required for inference. Non-Apple platforms support
> static analysis only.

```bash
git clone https://github.com/dawsonblock/TURB0.git
cd TURB0
pip install -e '.[apple]'
```

Non-Apple platform (CI, linting, static tests):

```bash
pip install -e '.[dev]'
```

Full development environment on Apple Silicon:

```bash
pip install uv nox
uv pip install -e '.[apple,dev]'
```

### Dependencies

| Package | Role | Platform |
|---|---|---|
| `mlx >= 0.30.0, < 1.0.0` | Core tensor ops + Metal execution | Apple Silicon only |
| `mlx-lm` (vendored v0.29.1) | LLM generation runtime | Apple Silicon only |
| `numpy` | Calibration data loading only (not in inference path) | Any |
| `nox` | Test matrix isolation | Any |
| `uv` | Fast package resolution | Any |

---

## Quick Start

### Option 1 — kwargs directly to `mlx_lm.generate`

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

response = generate(
    model,
    tokenizer,
    prompt="Explain KV-cache compression in one paragraph.",
    max_tokens=256,
    turboquant_k_start=0,           # token index to begin compression
    turboquant_k_bits=3,            # key quantization bit-width
    turboquant_group_size=64,       # quantization group size
    turboquant_rotation="hadamard",
    turboquant_residual_topk=2,     # residual error recovery components
    turboquant_v_bits=4,            # value quantization bit-width
    turboquant_v_enabled=True,
)
```

### Option 2 — manual cache upgrade after prefill

```python
from mlx_lm.models.cache import make_prompt_cache
from turboquant.integrations.mlx.upgrade import upgrade_cache_list
from turboquant.config import TurboQuantConfig

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
cache = make_prompt_cache(model)

# --- run prefill here ---

cfg = TurboQuantConfig(k_bits=3, k_group_size=64, rotation="hadamard")
events = upgrade_cache_list(cache, k_start=64, config=cfg)

for evt in events:
    print(f"Layer {evt.layer_index}: {evt.old_type} -> {evt.new_type} "
          f"(upgraded={evt.upgraded}, offset={evt.offset_at_upgrade})")

# Decode loop continues with compressed cache
```

### Option 3 — preset config

```python
from turboquant import TurboQuantConfig, TurboQuantPipeline

cfg = TurboQuantConfig.from_preset("balanced")  # high_compression | balanced | max_quality
pipeline = TurboQuantPipeline(cfg)
pipeline.build()
```

### Option 4 — offline calibration

```python
from turboquant.calibration import calibrate

calibrate(
    pipeline,
    data_loader,                           # any iterable of batches
    extract_kv=lambda b: (b["k"], b["v"]),
    mode="both",                           # "k", "v", or "both"
    max_batches=64,
)
# pipeline now uses fitted per-group scales → lower quantization error
```

---

## Configuration Reference

All behavior is controlled by a single `TurboQuantConfig` dataclass.

```python
from turboquant.config import TurboQuantConfig

cfg = TurboQuantConfig(
    k_bits=3,
    k_group_size=64,
    v_bits=4,
    v_group_size=64,
    v_enabled=True,
    rotation="hadamard",
    rotation_seed=42,
    rotation_pad_to_pow2=True,
    residual_mode="topk",
    residual_topk=2,
    block_tokens=256,
    return_mode="view",
)
cfg.validate()   # raises ValueError on invalid field combinations
```

### Fields

| Field | Type | Default | Description |
|---|---|:---:|---|
| `k_bits` | `int` | `3` | Key quantization bit-width (2–8). Lower = smaller, more distortion. |
| `k_group_size` | `int` | `64` | Tokens per quantization group for keys. |
| `v_bits` | `int` | `4` | Value quantization bit-width (2–8). |
| `v_group_size` | `int` | `64` | Tokens per quantization group for values. |
| `v_enabled` | `bool` | `True` | Enable value quantization. `False` = compress keys only. |
| `rotation` | `str` | `"hadamard"` | Whitening: `"hadamard"` · `"random_orthogonal"` · `"identity"`. |
| `rotation_seed` | `int` | `42` | Deterministic seed — must match encode and decode. |
| `rotation_pad_to_pow2` | `bool` | `True` | Pad head dim to next power-of-two for exact Hadamard. |
| `residual_mode` | `str` | `"qjl"` | Residual: `"topk"` · `"qjl"` · `"none"`. |
| `residual_topk` | `int` | `0` | Top-k residual components per group (0 = disabled). |
| `block_tokens` | `int` | `256` | Decode processing block size in tokens. |
| `return_mode` | `str` | `"view"` | Cache return: `"view"` (zero-copy) · `"copy"`. |

### Rotation modes

| Mode | Cost | Notes |
|---|---|---|
| `"hadamard"` | O(d²) | Best quality. Exact for power-of-two dims; orthogonal fallback otherwise. |
| `"random_orthogonal"` | O(d²) | QR-derived, deterministic via `rotation_seed`. |
| `"identity"` | O(1) | No whitening. Faster for large head dims, worse quality. |

### Residual modes

| Mode | Description |
|---|---|
| `"topk"` | Top-k largest-magnitude per-group errors stored as fp16 val + uint8 index. |
| `"qjl"` | Quantized Johnson-Lindenstrauss residual (default). |
| `"none"` | No residual error recovery. |

---

## Presets

```python
cfg = TurboQuantConfig.from_preset("balanced")   # recommended starting point
```

| Preset | k\_bits | v\_bits | residual\_topk | Memory | Quality |
|---|:---:|:---:|:---:|---|---|
| `"high_compression"` | 2 | 3 | 0 | Maximum reduction | Lowest quality |
| `"balanced"` | 3 | 4 | 2 | ~3.7× vs dense | Good — **recommended** |
| `"max_quality"` | 4 | 8 | 4 | ~3.0× vs dense | Near-lossless |

Start with `"balanced"`, then tune `k_bits` / `residual_topk` using `make certify-apple-runtime`.

---

## API Reference

### Package exports

```python
import turboquant

turboquant.TurboQuantConfig        # Config dataclass
turboquant.TurboQuantPipeline      # Encode/decode pipeline
turboquant.TurboQuantKVCache       # Full KV cache (keys + values)
turboquant.TurboQuantKCache        # Keys-only mlx_lm adapter
turboquant.KVCompressor            # Alias for TurboQuantKVCache
turboquant.calibrate               # Offline scale calibration
turboquant.upgrade_cache_list      # Canonical cache upgrade
turboquant.__version__             # "0.2.2"

# Environment helpers
turboquant.has_mlx()               # bool — MLX importable?
turboquant.is_apple_silicon()      # bool — arm64 Darwin?
turboquant.require_mlx()           # raises ImportError if MLX unavailable
turboquant.check_mlx_version()     # raises if MLX outside [0.30.0, 1.0.0)
```

### `upgrade_cache_list`

```python
from turboquant.integrations.mlx.upgrade import upgrade_cache_list

events = upgrade_cache_list(
    cache_list,    # list[KVCache] from make_prompt_cache(model)
    k_start=64,    # compress keys from this token offset onward
    config=cfg,    # TurboQuantConfig
)
# Returns list[CacheUpgradeEvent]
```

`CacheUpgradeEvent` fields:

| Field | Type | Description |
|---|---|---|
| `layer_index` | `int` | Layer that was processed |
| `upgraded` | `bool` | Whether this layer was upgraded |
| `old_type` | `str` | Cache class name before upgrade |
| `new_type` | `str` | Cache class name after upgrade |
| `offset_at_upgrade` | `int` | Token offset at which upgrade occurred |

`upgrade_cache_list` is **idempotent** — layers already using `TurboQuantKCache` are skipped.
Only model families in the certified allowlist (`llama`, `gemma`) are upgraded; others are left
as-is and logged.

### `TurboQuantKCache`

The mlx\_lm-protocol-compatible per-layer cache adapter:

```python
from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache
from turboquant.config import TurboQuantConfig

cache = TurboQuantKCache(
    TurboQuantConfig(k_bits=3, k_group_size=64, rotation="hadamard",
                     v_bits=4, v_enabled=True)
)
# Implements: update_and_fetch(keys, values), .state, .offset
```

### State schema

```python
state = cache.state()
assert state["schema_version"] == 2

from turboquant.runtime.state import validate_state
validate_state(state)   # raises on structural mismatch
```

---

## Evaluation

```python
from turboquant.eval import perplexity_report, drift_report, memory_report
from turboquant.config import TurboQuantConfig

cfg = TurboQuantConfig.from_preset("balanced")

# Perplexity delta vs dense  —  quality gate: delta_ppl <= 0.5
ppl = perplexity_report(model, input_ids, turboquant_config=cfg)
# {"dense_ppl": 12.3, "tq_ppl": 12.6, "delta_ppl": 0.3, "n_tokens": 63}

# Logit KL divergence  —  quality gate: mean_kl <= 0.1
drift = drift_report(model, input_ids, turboquant_config=cfg)
# {"mean_kl": 0.004, "max_kl": 0.021, "n_tokens": 63}

# Cache memory comparison
mem = memory_report(model, input_ids, turboquant_config=cfg)
# {"dense_cache_bytes": 2097152, "tq_cache_bytes": 524288, "ratio": 4.0}
```

Quality gates enforced by `make certify-apple-runtime`:

| Gate | Threshold | Enforced by |
|---|---|---|
| Perplexity delta | Δppl ≤ 0.5 | `run_quality_eval.py` |
| Mean KL divergence | mean\_kl ≤ 0.1 | `run_quality_eval.py` |

See [docs/evaluation.md](docs/evaluation.md) for threshold interpretation.

---

## Testing

```bash
# Platform-agnostic, no MLX required  (~1 second)
make test-static

# MLX unit tests — Apple Silicon only
make test-mlx

# Structural integration tests — no model weights  (~2 seconds)
make test-structural

# Path-proof and offset-tracking structural tests
make test-path-proof

# Model smoke tests — skip cleanly when env vars are absent
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
make test-smoke-llama     # end-to-end Llama generation with TurboQuant active
make test-long-context    # long-context (>256 tokens) — reuses TQ_TEST_LLAMA_MODEL
```

### Test matrix

| Target | Directory | Needs MLX | Needs weights |
|---|---|:---:|:---:|
| `test-static` | `tests/unit_static/` | ✗ | ✗ |
| `test-mlx` | `tests/unit_mlx/` + `tests/integration_mlx/` | ✓ | ✗ |
| `test-structural` | `tests/integration_mlx/` (3 structural files) | ✓ | ✗ |
| `test-path-proof` | `tests/integration_mlx/test_path_not_dense_fallback.py` | ✓ | ✗ |
| `test-smoke-llama` | `tests/integration_mlx/test_llama_runtime_smoke.py` | ✓ | ✓ |
| `test-smoke-gemma` | `tests/integration_mlx/test_gemma_runtime_smoke.py` | ✓ | ✓ |
| `test-long-context` | `tests/integration_mlx/test_long_context_stability.py` | ✓ | ✓ |

### Tests requiring model weights

```bash
# Llama (start here — smaller model; certify before Gemma)
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
make test-smoke-llama    # asserts: tokens generated > 0, TQ active, no dense fallback
make test-long-context   # asserts: no NaN logprobs on prompts > 256 tokens

# Gemma (run only after Llama smoke is artifact-backed)
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"
make test-smoke-gemma
```

Without these variables, all three smoke tests skip automatically with a clear message.

### Runtime certification

```bash
make certify-structural       # no weights; produces JUnit XML
make certify-apple-runtime    # full certification (requires weight env vars above)
```

See [docs/validation-local.md](docs/validation-local.md) for the full walkthrough.

---

## Benchmarks

```bash
python benchmarks/exploratory/bench_memory_footprint.py    # bit-width x seq-len table
python benchmarks/exploratory/bench_dense_vs_turboquant.py # paired latency comparison
python benchmarks/exploratory/bench_decode_streaming.py    # streaming attention throughput
python benchmarks/exploratory/bench_decode_step.py         # single decode step
python benchmarks/exploratory/bench_k_encode.py            # K-encode micro-benchmark
```

### Synthetic micro-benchmarks (Apple Silicon, Python 3.10.12)

> **NOTE:** The figures below are from early development and have not been
> reproduced against saved certification artifacts. They are illustrative only.

| Benchmark | Result |
|---|---|
| K-Encode (`encode_k_block`, shape [1, 32, 128, 128]) | **0.10 ms / step** |
| Decode step (`append_keys`, 1 new token) | **0.03 ms / step** |

### Paired generative benchmarks — 64 tokens, 30 total runs

> **NOTE:** Historical figure; not reproduced against saved certification artifacts.

| Mode | Avg latency | Tok/s | Correctness |
|---|---|---|---|
| Dense | 0.52 s | 147–163 | ✅ All 15 passed |
| TurboQuant | 6.80 s | 9–10 | ✅ All 15 passed — output identical to dense |

> **Speed note:** Current decode speed is uncompiled Python-level MLX. Use `mx.compile(inner)` for
> ~2× speedup. Fused Metal kernel (`TQ_USE_METAL=1`) is much faster but experimental.

---

## Model Support Matrix

| Architecture | Status | Notes |
|---|:---:|---|
| **Llama** (Llama 2, Llama 3, TinyLlama) | ⬜ Wired, uncertified | Smoke test wired — set `TQ_TEST_LLAMA_MODEL` to activate |
| **Gemma** (Gemma 2) | ⬜ Wired, uncertified | Smoke test wired — set `TQ_TEST_GEMMA_MODEL` (run Llama first) |
| Qwen | ⬜ Exploratory | Auto-routed via SDPA dispatch; uncertified |
| Mistral | ⬜ Exploratory | Auto-routed via SDPA dispatch; uncertified |
| Phi | ⬜ Exploratory | Auto-routed via SDPA dispatch; uncertified |
| Any other `base.py` model | ⬜ Auto-routed | Automatic if model uses `scaled_dot_product_attention` from `base.py` |

Adding a new architecture to the **wired allowlist** requires no per-model code changes.
See [docs/integration.md](docs/integration.md).

---

## Hardware Requirements

| | |
|---|---|
| **Platform** | macOS · Apple Silicon (M1 / M2 / M3 / M4) |
| **Python** | ≥ 3.9 |
| **MLX** | ≥ 0.30.0, < 1.0.0 |
| **mlx-lm** | Vendored v0.29.1 (see [VENDORED_MLX_LM.md](VENDORED_MLX_LM.md)) |
| **Non-Apple** | Static tests, linting, and type-checking only |

---

## Project Layout

```
TurboQuantX1/
├── turboquant/                   Core library
│   ├── __init__.py               Lazy-import entry point (MLX-free until first use)
│   ├── _deps.py                  has_mlx() / is_apple_silicon() / require_mlx()
│   ├── config.py                 TurboQuantConfig — single source of truth
│   ├── errors.py                 TurboQuantError hierarchy
│   ├── core/                     Quantization primitives
│   │   ├── rotation.py           FixedRotation (Hadamard / QR / identity)
│   │   ├── quantizer.py          GroupScalarQuantizer + vectorized pack/unpack
│   │   ├── residual.py           encode_topk_residual / decode_topk_residual
│   │   └── pipeline.py           encode_k_block / decode_k_block
│   ├── runtime/                  Inference runtime
│   │   ├── layout.py             ensure_layout [B, H, T, D]
│   │   ├── kv_interface.py       TurboQuantKVCache + TurboQuantKeysView
│   │   ├── attention.py          turboquant_streaming_attention
│   │   ├── support.py            Model family allowlist
│   │   └── state.py              STATE_SCHEMA_VERSION + validate_state()
│   ├── integrations/mlx/
│   │   ├── cache_adapter.py      TurboQuantKCache (mlx_lm protocol adapter)
│   │   └── upgrade.py            upgrade_cache_list() — canonical upgrade API
│   ├── calibration/
│   │   └── fit_quantizer.py      calibrate() — offline scale fitting
│   ├── eval/
│   │   ├── perplexity.py
│   │   ├── generation_drift.py
│   │   └── memory.py
│   └── kernels/                  Metal kernel dispatch (experimental)
├── mlx_lm/                       Patched vendored mlx-lm v0.29.1
│   ├── models/base.py            SDPA — TurboQuantKeysView auto-dispatch
│   ├── models/llama.py           Llama wiring
│   ├── models/gemma.py           Gemma wiring
│   └── generate.py               maybe_turboquant_k_cache + generate_step
├── tests/
│   ├── unit_static/              Platform-agnostic import + version tests
│   ├── unit_mlx/                 MLX unit tests (Apple Silicon)
│   └── integration_mlx/          Integration tests (Apple Silicon)
├── benchmarks/
│   ├── exploratory/              Micro-benchmarks and latency scripts
│   └── runtime_cert/             Certification harness
├── scripts/
│   ├── certify_apple_runtime.sh
│   ├── preflight.py
│   └── run_benchmarks.sh
├── docs/
├── pyproject.toml
├── Makefile
└── noxfile.py
```

---

## Component Status

| Component | Status |
|---|:---:|
| `TurboQuantKVCache` | ✅ 38 / 38 tests |
| `encode_k_block` / `decode_k_block` pipeline | ✅ Single path, zero runtime branches |
| `FixedRotation` (Hadamard / QR / identity) | ✅ Deterministic, save / load |
| `GroupScalarQuantizer` + offline calibration | ✅ Dynamic + calibrated modes |
| Top-k sparse residual | ✅ Per-group, configurable k |
| Pure-MLX bit-packing | ✅ Vectorized, no NumPy sync |
| Versioned state schema (`schema_version: 2`) | ✅ `validate_state()` enforced |
| `TurboQuantKCache` mlx\_lm adapter | ✅ 20 / 20 tests |
| Streaming attention | ✅ `turboquant.runtime.attention` |
| Centralized SDPA dispatch (`base.py`) | ✅ All model families auto-routed |
| Gemma streaming attention | ⬜ Wired, uncertified |
| Llama streaming attention | ⬜ Wired, uncertified |
| `upgrade_cache_list` | ✅ Canonical, idempotent |
| Eval suite (perplexity / KL / memory) | ✅ `turboquant.eval` |
| Quality gates (Δppl ≤ 0.5, mean\_kl ≤ 0.1) | ✅ `run_quality_eval.py` |
| MLX version bounds `[0.30.0, 1.0.0)` | ✅ Enforced at import |
| NaN / overflow guards | ✅ Encode + attention |
| Structural proof tests | ✅ 8 tests — offset tracking, cache independence, round-trip, streaming attention |
| Model smoke tests (Llama / Gemma / long-context) | ⬜ 3 conditional — skip until `TQ_TEST_LLAMA_MODEL` / `TQ_TEST_GEMMA_MODEL` set |
| Static unit tests | ✅ 10 / 10 modules |
| Fused Metal kernel | ✅ `TQ_USE_METAL=1` (experimental) |
| `mx.compile` JIT fallback | ✅ ~2× speedup |
| Perplexity at production scale | ⬜ Not yet measured |

---

## Limitations

- **Quality gates on small inputs** — `run_quality_eval.py` enforces Δppl ≤ 0.5 and
  mean\_kl ≤ 0.1 but these are validated on synthetic or small-model data. Run
  `make certify-apple-runtime` with real model weights to validate on your target model.

- **Hadamard is O(d²)** — for very large head dimensions consider `rotation="identity"` to
  avoid rotation overhead at the cost of marginally worse compression quality.

- **Metal kernel is experimental** — the fused decode+dequant Metal kernel requires
  `TQ_USE_METAL=1`. The default path uses `mx.compile()` (~2× speedup over naive MLX).

- **Only Llama and Gemma are in the wired allowlist** — other architectures auto-route via the centralized
  SDPA dispatch but are uncertified. [Adding to the allowlist](docs/integration.md) is a
  one-function change.

- **Apple Silicon only for inference** — MLX does not install on non-Apple platforms. All
  inference, quantization, attention, and calibration code requires Apple Silicon.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Component map, data-flow, memory model |
| [docs/cache-format.md](docs/cache-format.md) | State dict schema v2, uint32 packing layout |
| [docs/integration.md](docs/integration.md) | Step-by-step guide for adding new model families |
| [docs/evaluation.md](docs/evaluation.md) | Metrics reference, benchmark workflow, thresholds |
| [docs/supported-surface.md](docs/supported-surface.md) | Certified surface definition |
| [docs/validation-local.md](docs/validation-local.md) | Local validation walkthrough |
| [docs/support_matrix.md](docs/support_matrix.md) | Model × platform support matrix |
| [docs/benchmark_methodology.md](docs/benchmark_methodology.md) | Benchmark reproducibility contract |
| [VENDORED_MLX_LM.md](VENDORED_MLX_LM.md) | mlx-lm vendoring boundary and patch notes |
| [RELEASE_CANDIDATE_NOTES.md](RELEASE_CANDIDATE_NOTES.md) | v0.2.2 release notes |

---

## Development Setup

```bash
pip install uv nox
uv pip install -e '.[apple,dev]'    # Apple Silicon
uv pip install -e '.[dev]'          # Non-Apple  (static checks only)
```

### Makefile reference

| Target | Description |
|---|---|
| `make help` | Print all available targets |
| `make install-dev` | Install editable package with dev extras |
| `make install-apple` | Install editable with Apple Silicon + dev extras |
| `make compile` | Compile all source and test modules |
| `make lint` | Ruff linting + formatting (via nox) |
| `make typecheck` | Mypy type validation (via nox) |
| `make test` | Alias for `test-static` |
| `make test-static` | Platform-agnostic static tests |
| `make test-mlx` | MLX unit tests (Apple Silicon) |
| `make test-structural` | Integration tests — no model weights |
| `make test-path-proof` | Verify TQ path is active; offset-tracking and cache-independence proofs |
| `make test-smoke-llama` | Llama runtime smoke — requires `TQ_TEST_LLAMA_MODEL` |
| `make test-smoke-gemma` | Gemma runtime smoke — requires `TQ_TEST_GEMMA_MODEL` |
| `make test-long-context` | Long-context stability (>256 tokens) — requires `TQ_TEST_LLAMA_MODEL` |
| `make certify-structural` | Structural cert (JUnit XML output) |
| `make certify-apple-runtime` | Full certification with model weights |
| `make build-dist` | Build wheel + sdist |
| `make validate-apple` | Apple Silicon runtime validation |
| `make clean` | Remove build artifacts |

---

## Contributing

1. Fork and create a feature branch.
2. Run `make test-static` — passes on any platform.
3. On Apple Silicon run `make test-mlx` and `make test-structural`. If you have a supported model, also run `make test-smoke-llama`.
4. Run `make lint` and `make typecheck`.
5. Open a pull request with a clear description and any benchmark results.

**Adding a new model family to the certified allowlist** — the centralized SDPA dispatch means
no per-model code changes are required for basic support. See [docs/integration.md](docs/integration.md).

**Changing the state schema** — increment `STATE_SCHEMA_VERSION` in
`turboquant/runtime/state.py` and update `validate_state()` to handle both old and new versions.

---

## License

[MIT](LICENSE)
