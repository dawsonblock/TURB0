# TurboQuant Architecture

> **Status**: prototype, not production-ready  
> **Last updated**: April 2026

---

## 1. Overview

TurboQuant compresses the KV (key-value) cache of transformer language models
by quantising both K and V heads to low bit-widths (typically 3–4 bits) and
decompressing on-the-fly during attention. The goal is to cut memory bandwidth
at decode time on Apple Silicon (MLX backend). This document is an operational
description of the current prototype, not a claim of broad support or runtime certification.

The supported runtime story in the current repo is intentionally narrow.  The
canonical path is:

`mlx_lm.generate.generate_step` → `_infer_model_family(...)` →
`maybe_turboquant_k_cache(...)` → `upgrade_cache_list(...)` →
`TurboQuantKCache.update_and_fetch(...)` → `TurboQuantKeysView` →
`mlx_lm.models.base.scaled_dot_product_attention(...)` →
`turboquant_streaming_attention(...)`

Lower-level helpers that construct `TurboQuantKCache` directly still exist for
evaluation and compatibility, but they are secondary surfaces and not the
supported public entry point. If this document disagrees with the allowlist or
the upgrade gate, the code wins.

```text
Input token
     │
     ▼
┌────────────────────────────────────────────────┐
│                   Model Layer                  │
│                                                │
│  Linear projections  →  Q, K, V               │
│       │                   │                    │
│       │           ┌────────────────────┐       │
│       │           │  TurboQuantKCache  │       │
│       │           │                    │       │
│       │           │                   │       │
│       │           │  encode_k()  encode_v()│       │
│       │           │  k_packed   v_packed │       │
│       │           └──────────┬─────────┘       │
│       │                      │ iter_blocks()    │
│       ▼                      ▼                 │
│  (q rotated in score_block) → block attention  │
│                      │                         │
│                      ▼                         │
│                   Output                       │
└────────────────────────────────────────────────┘
```text
---

## 2. Package structure

```text
turboquant/
├── config.py               # TurboQuantConfig dataclass (algorithm, presets, effective_bits)
├── core/
│   ├── pipeline.py         # TurboQuantPipeline — per-layer quantise/dequantise; owns rotation
│   ├── quantizer.py        # LloydMaxScalarQuantizer (paper-faithful) + GroupScalarQuantizer
│   ├── polar_quant.py      # PolarQuantizer — recursive polar transform (arXiv:2502.02617)
│   ├── rotation.py         # FixedRotation — hadamard / identity / random; apply/invert
│   ├── residual_codec.py   # build_residual_codec() — algorithm-aware dispatch
│   └── qjl.py              # QJLProjector — 1-bit sketch; estimate_inner_product()
├── runtime/
│   ├── kv_interface.py     # TurboQuantKVCache — paper_kv / k_only storage modes
│   ├── attention.py        # turboquant_streaming_attention — V decode via v_blocks
│   ├── support.py          # model-family allowlist gate for canonical upgrade path
│   ├── events.py           # JSONL persistence-side upgrade / failure events
│   └── state.py            # STATE_SCHEMA_VERSION + validate_state()
├── eval/
│   ├── __init__.py
│   ├── perplexity.py       # perplexity_report — dense vs TQ comparison
│   ├── generation_drift.py
│   └── memory.py
└── tests/                  # compat stub (canonical tests live in tests/)
```text
```text
mlx_lm/
├── models/
│   ├── cache.py            # KVCache, TurboQuantKCache (adapter), helpers
│   └── gemma.py            # Gemma attention wired to streaming attention
├── generate.py             # generate_step
└── upgrade.py              # (in integrations/mlx) canonical upgrade entry point

```text
---

## 3. Key components

### 3.1 TurboQuantConfig (`turboquant/config.py`)

Production configuration dataclass.  Fields:

| field | default | description |
|---|---|---|
| `k_bits` | 3 | bits per key element |
| `k_group_size` | 64 | keys quantised in groups of this size |
| `v_bits` | 4 | bits per value element |
| `v_group_size` | 64 | values quantised in groups of this size |
| `v_enabled` | True | whether to quantise V (K is always quantised) |
| `algorithm` | `"turboquant_prod"` | **New**: `"turboquant_mse"` (rotate + Lloyd-Max, no residual) or `"turboquant_prod"` (MSE + 1-bit QJL residual) |
| `rotation` | `"hadamard"` | pre-rotation: `"identity"`, `"hadamard"`, `"random_orthogonal"` |
| `residual_mode` | `"qjl"` | residual codec: `"none"` (MSE only), `"qjl"` (1-bit QJL), `"topk"` (sparse top-k, experimental) |
| `quantizer_mode` | `"scalar"` | main quantiser: `"scalar"` (LloydMax/GroupScalar) or `"polar"` (PolarQuantizer) |
| `residual_topk` | 0 | top-k residual elements per group (topk mode only) |
| `block_tokens` | 256 | compatibility-only knob; accepted by `TurboQuantConfig` but not currently used in the attention dispatch path |
| `eps` | 1e-6 | numerical stability floor |
| `scale_dtype` | `"float16"` | dtype for scale factors |
| `v_scale_dtype` | `"float16"` | dtype for V scale factors |

**Convenience helpers** added in §paper alignment:

| method | description |
|---|---|
| `is_mse_mode()` | True when `algorithm == "turboquant_mse"` |
| `is_prod_mode()` | True when `algorithm == "turboquant_prod"` |
| `effective_bits_per_channel_k(d)` | Effective K bpc using paper §3 formula |
| `effective_bits_per_channel_v(d)` | Effective V bpc |
| `effective_bits_per_channel_total(d)` | Average (K + V) bpc |
| `from_preset(name)` | Named presets: `"paper_mse"`, `"paper_prod"`, `"balanced"`, … |

**Algorithm–residual contract** (enforced by `validate()`):
- `turboquant_mse` → `residual_mode` must be `"none"`
- `turboquant_prod` → `residual_mode` must be `"qjl"` or `"topk"` (experimental)

> **Legacy note**: `mlx_lm.models.cache.TurboQuantConfig` uses old field names
> (`main_bits`, `group_size`, `return_mode`, …).  It is a shim that maps to the
> production dataclass — see [integration.md](integration.md).

### 3.2 TurboQuantKVCache (`turboquant/runtime/kv_interface.py`)

The core compressed K/V store and the canonical runtime storage engine.
`TurboQuantKCache` is a thin internal `mlx_lm` adapter around this type when
`_BaseCache` compatibility is needed.

**Constructor:**
```python
TurboQuantKVCache(
    config: TurboQuantConfig,          # positional or keyword
    *,
    quantize_main=None,                # optional — auto-selects based on algorithm
    dequantize_main=None,              # optional — auto-selects based on algorithm
)
```
If `quantize_main` / `dequantize_main` are omitted:
- **MSE or Prod mode** (`is_mse_mode()` or `is_prod_mode()`) → `LloydMaxScalarQuantizer` (paper-faithful)
- **Legacy / experimental** → `GroupScalarQuantizer`

**Storage modes** (`storage_mode` attribute):
- `"paper_kv"` — K and V both encoded; V stored in `v_blocks` via `encode_k_block`; dense `v_cache` is empty. Used for MSE/Prod configs.
- `"k_only"` — K compressed; V stored dense in `v_cache`. Legacy behaviour.

**Core lifecycle:**

1. `append_keys(k)` — encode and store one key block
2. `update_and_fetch(k, v)` → `(TurboQuantKeysView, v)` — MLX-LM adapter
   protocol: appends keys (compressed) and values (dense), returns a view
   for streaming attention
3. `iter_blocks()` — yield `EncodedKeyBlock` objects for streaming attention
4. `decode_block_full(index)` → decoded `mx.array` for a specific block
5. `state()` / `from_state(state, *, quantize_main, dequantize_main)` — serialise/restore (schema v2)
6. `clear()` — reset all buffers (blocks, v_cache, offsets)

**Convenience properties:**

| property / method | description |
|---|---|
| `nbytes` | total compressed bytes (alias for `byte_size()`) |
| `k_packed` | `packed_main` tensor of the first block, or `None` |
| `v_cache` | list of dense value tensors appended via `update_and_fetch` |
| `num_blocks` | number of stored key blocks |
| `memory_breakdown()` | dict: `k_packed_main`, `k_scales`, `v_dense`, `total` bytes |

**Runtime surfaces**

| surface | status | notes |
|---|---|---|
| `upgrade_cache_list(...)` | Canonical runtime path | Enforces the `model_family` allowlist before mutating caches |
| Direct `TurboQuantKCache(...)` construction | Internal / eval only | Used by comparison helpers and compatibility shims; bypasses Gate 2 |
| `KVCache.to_turboquant()` and `_collect_logits_compressed()` | Secondary helpers | Intentional bypasses for eval / compatibility, not the supported public path |

### 3.3 Streaming attention (`turboquant/runtime/attention.py`)

`turboquant_streaming_attention(queries, keys_view, *, scale)`:

- Rotates queries _per block_ inside `score_block()` via `FixedRotation.apply()`;
  query rotation is applied inside each block, not at the top-level call site
- Iterates over K/V blocks with `iter_blocks()` / `decode_block_full()`
- Accumulates per-block score tensors in a Python list, then concatenates them
  with `mx.concatenate` and applies a single standard `mx.softmax` across the
  full key dimension.  The full score vector is materialised before softmax.
  (The implementation does _not_ use a streaming log-sum-exp approach.)
- Supports both `TurboQuantKCache` (via `._impl`) and `TurboQuantKVCache`
  (direct) through `impl = getattr(cache, "_impl", cache)` dispatch

`maybe_turboquant_attention(q, k, v, mask, scale, fallback, cache)`:
- **Legacy helper** — predates the centralized `base.py` SDPA type-guard. Used only as a
  fallback for custom attention paths that do not go through `base.py`'s
  `scaled_dot_product_attention`. For all `base.py`-routed models the SDPA type-guard handles
  dispatch automatically.
- Dispatches: if `isinstance(k, TurboQuantKeysView)` → streaming path;
  else → `fallback(q, k, v, mask, scale)`

### 3.4 Rotation (`turboquant/core/rotation.py`)

Three modes:

| mode | description | cost |
|---|---|---|
| `"identity"` | no rotation (fastest, least entropy spreading) | O(1) |
| `"hadamard"` | dense Hadamard matrix via NumPy → `mx.array` | O(d²) |
| `"random_orthogonal"` | random orthogonal via SVD at init | O(d²) |

> **Note**: the Hadamard implementation uses a dense matrix multiply, not the
> fast Walsh-Hadamard transform butterfly (O(d log d)).

### 3.5 PolarQuantizer (`turboquant/core/polar_quant.py`)

Implements the PolarQuant algorithm (arXiv:2502.02617, Zandieh et al., AISTATS 2026) as a
drop-in replacement for `GroupScalarQuantizer`.

**Algorithm:**
1. **Random preconditioning** — already applied by `rotation.py` (Hadamard rotation makes angles analytically tractable).
2. **Recursive polar transform** (L = 4 levels):
   - Level 1: pair coordinates `(x[2j], x[2j+1])` → angle ∈ [0, 2π), radius ∈ ℝ₊.
   - Levels 2–4: pair radii → angle ∈ [0, π/2), new radius.
3. **Angle quantisation**:
   - Level 1: **4 bits** (uniform distribution → Lloyd-optimal 16-centroid codebook).
   - Levels 2–4: **2 bits each** (concentrated distribution near π/4 → Lloyd-optimal 4-centroid codebook).
   - Final `d/2^L` radii stored as **float16** (no group scale factors).

**Memory** for d = 128, L = 4: 4·64 + 2·32 + 2·16 + 2·8 = 368 angle bits + 128 radii bits = 496 bits → **3.875 bits/dim, zero scale overhead**.  Scalar 3-bit with group=64 costs 3·128 + 16·2 = 416 bits = **3.25 bits/dim** but carries float16 scale storage.

**Benchmark results (Apple Silicon M-series, April 2026, d=128, T=512–1024 — backed by `artifacts/benchmarks/polar_vs_scalar.json`):**

| metric | GroupScalar 3-bit | PolarQuant |
|---|---|---|
| bits/dim | 3.25 + scale overhead | **3.875 (zero overhead)** |
| MSE | ~0.064 | **~0.038 (40% lower)** |
| encode latency | ~0.4 ms | **~0.04 ms (10× faster)** |
| decode latency | ~0.2 ms | ~0.4 ms |

Activate via `TurboQuantConfig(quantizer_mode="polar")`.  When `quantizer_mode="polar"`, `EncodedKeyBlock.polar` carries the `PolarQuantPayload`; `packed_main` and `scales` are `None` and no residual correction is applied.

**API** (identical to `GroupScalarQuantizer`):
```python
pq = PolarQuantizer(n_levels=4, bits_l1=4, bits_le=2)
payload = pq.encode(x)          # → PolarQuantPayload
x_hat  = pq.decode(payload)     # → mx.array

# pipeline adapters
packed, scales = pq.quantize(x, config=config)  # scales is None
x_hat = pq.dequantize(packed, scales, config=config)
```

### 3.6 TurboQuantPipeline (`turboquant/core/pipeline.py`)

Wraps the per-layer encode/decode primitives. Now features an explicit `.build()` phase that pre-allocates caches, quantizers, and fixed rotations ahead-of-time to avoid any branch-heavy lazy initialization during hot-path execution.

### 3.7 State schema (`turboquant/runtime/state.py`)

State dicts carry `schema_version: 2`.  `validate_state(state, config)` checks:

- `schema_version` present and equal to `STATE_SCHEMA_VERSION`
- Required scalar keys: `offset`, `d_head`, `d_pad`, `v_dim`, `v_pad`
- Token dimension of `k_packed` ≥ `offset`
- Group count consistent with config

### 3.8 Event architecture

TurboQuant currently has a documented split event model rather than a unified
runtime event bus:

- `turboquant.integrations.mlx.upgrade.CacheUpgradeEvent` is the lightweight
  runtime result returned by `upgrade_cache_list(...)`.
- `turboquant.runtime.events.CacheUpgradeEvent`, `UpgradeFailureEvent`, and
  `EventLog` are persistence-layer types used to write `events.jsonl` during
  certification and offline analysis.
- The canonical decode path does not automatically persist runtime upgrade
  events. JSONL logging is a secondary certification surface, not part of the
  default decode loop.

---

## 4. Data flow: one decode step

```text
q [B, H_q, 1, d]    k [B, H_kv, 1, d]    v [B, H_kv, 1, d]
        │                    │                    │
        │            TurboQuantKCache.update_and_fetch(k, v)
        │                    │
        │            encode_k() → k_packed  [B, H, T, n_words] uint32
        │            encode_v() → v_packed  [B, H, T, n_words] uint32
        │                    │
        │            TurboQuantKeysView (lazy proxy)
        │                    │
        └──── turboquant_streaming_attention(q, keys_view) ─────────────┐
                             │                                           │
              score_block() per stored block                             │
                   ┌─────────┴──────────┐                               │
                   │   for each block   │                                │
                   │   FixedRotation.apply(q) → q_rot  (per block)      │
                   │   decode_k() → k_blk  (rotated)                    │
                   │   decode_v() → v_blk                               │
                   │   scores = q_rot @ k_blk.T / scale                 │
                   └───────────────────┘                                │
                             │  (list of per-block score tensors)        │
                         mx.concatenate → full_scores                    │
                         mx.softmax(full_scores)                         │
                             │                                           │
                          output [B, H_q, 1, d] ◄────────────────────────┘
```text
---

## 5. Memory model

For a sequence of T tokens, N KV heads, head dimension d, at b bits/group of g:

$$\text{bytes}_{K}^{\text{scalar}} \approx \frac{b \cdot N \cdot T \cdot d}{8} + \frac{2 \cdot N \cdot T \cdot d}{g \cdot 8} \cdot \text{sizeof}(\text{scale\_dtype})$$

For PolarQuant (L levels, bits_l1 = 4, bits_le = 2):

$$\text{bits/dim}_{\text{polar}} = \frac{\text{bits\_l1} \cdot d/2 + \text{bits\_le} \cdot \sum_{\ell=2}^{L} d/2^\ell + 16 \cdot d/2^L}{d}$$

At L = 4, d = 128: 3.875 bits/dim with **no scale storage**.

The V component uses scalar quantisation in both modes.  At 3-bit K (scalar) + 4-bit V with group=64 and float16 scales, TurboQuant uses roughly **7–9×** less memory than float16 dense K for sequences > 256 tokens. Measured compression ratios (Apple Silicon, April 2026):

| quantiser | bits | group | bytes/token | vs dense |
|---|---|---|---|---|
| Scalar | 4 | 64 | 272 | 7.5x |
| Scalar | 3 | 64 | 240 | 8.5x |
| Scalar | 2 | 64 | 144 | 14.2x |
| Scalar | 3 | 32 | 256 | 8.0x |
| **PolarQuant** | **~3.875** | **—** | **~248** | **~8.3x** |

See `benchmarks/exploratory/bench_memory_footprint.py` and `artifacts/benchmarks/memory_footprint.txt`.

---

## 6. Limitations

- **Hadamard** is O(d²) — not the butterfly O(d log d).  For large d (≥ 128) this adds noticeable encode overhead.
- **Residual** is top-k sparse; the legacy sign-sketch residual is not supported in the production path.
- **V quantisation** is enabled by default (`v_enabled=True`).  Family-specific
  integrations may disable it explicitly when V quantisation degrades quality on
  a given model.  There is no family-level override in the core config; callers
  must pass `v_enabled=False` explicitly.
- **Direct adapter construction remains available** — `TurboQuantKCache(...)`
  and compatibility helpers exist for eval and interop, but they bypass the
  canonical `upgrade_cache_list(...)` support gate.
- **Llama and Gemma** wiring is complete (see [integration.md](integration.md)). Other model
  families are not supported — `upgrade_cache_list` raises `UnsupportedModelError` for families
  not in `SUPPORTED_FAMILIES`. Adding support requires both editing the allowlist and wiring the
  model; SDPA dispatch routing is not sufficient on its own.
- **Event layers are bifurcated** — runtime upgrade decisions and JSONL
  persistence use different event types. The split is documented; it is not yet
  a unified live event pipeline.


## 7. Validation boundary

This repository now includes packaging metadata and public static CI, but that CI does not certify MLX runtime behavior. Real runtime validation still requires an Apple Silicon Mac with `mlx` installed. Use `scripts/validate_apple_silicon.sh` for the supported local validation path.

## 8. Experimental & Native Acceleration
TurboQuant is a research-grade KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation for selected Llama-family and Gemma-family models. Custom Metal kernels are experimental and not part of the default supported runtime.

- **MLX JIT Core Compilation (Supported Default)**: Fallback stream topologies (`dequantize_groups`, `decode_k_fallback`) aggressively compile inline Python iterators directly into static C++ execution trees (`mx.compile(fn, shapeless=False)`). Configuration constants (e.g., bit-widths, padding sizes) are aggressively injected at the compilation cache layer `_DEQUANT_CACHE[key]` effectively stripping dispatch latency.
- **Metal Shader Injection (Experimental)**: Raw logic is ported directly to Apple Silicon via explicit `mx.fast.metal_kernel` C++ templates (`decode_k.metal`). Heavy inner accumulation loops (`resid_idx` matches) are natively untrolled utilizing statically mapped compilation signatures `[("BITS", config.k_bits)]`.

