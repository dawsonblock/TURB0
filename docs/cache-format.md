# TurboQuant Cache Format

> **Schema version**: 2  
> **Defined in**: `turboquant/runtime/state.py`

---

## 1. State dict (v2)

`KVCompressor.state()` returns a Python dict with the following keys:

| key | type | description |
| --- | --- | --- |
| `schema_version` | `int` | always `2` |
| `offset` | `int` | number of tokens stored |
| `d_head` | `int` | original (un-padded) head dimension |
| `d_pad` | `int` | padded head dimension (multiple of group_size) |
| `v_dim` | `int` | original value dimension |
| `v_pad` | `int` | padded value dimension |
| `k_packed` | `mx.array` uint32 | packed K codes `[B, H, offset, n_words]` |
| `k_scales` | `mx.array` float16 | K scale factors `[B, H, offset, n_groups]` |
| `k_resid_vals` | `mx.array` or `None` | top-k residual values `[B, H, offset, topk]` |
| `k_resid_idx` | `mx.array` or `None` | top-k residual indices `[B, H, offset, topk]` |
| `v_packed` | `mx.array` uint32 or `None` | packed V codes |
| `v_scales` | `mx.array` float16 or `None` | V scale factors |

### Packed format

K codes are packed into uint32 words.  For `k_bits = b` and (padded) head
dimension `d_pad`:

```text
codes_per_word = floor(32 / b)
n_words        = ceil(d_pad / codes_per_word)
k_packed.shape = [B, H, T, n_words]
```

Each uint32 stores `codes_per_word` b-bit unsigned integers in the **low bits
first** order:

```text
word = code[0] | (code[1] << b) | (code[2] << 2b) | ...
```

Unpacking extracts each code as `(word >> (i * b)) & ((1 << b) - 1)`.

### Scale format

One scale per group:

```text
n_groups      = ceil(d_pad / group_size)
k_scales.shape = [B, H, T, n_groups]
```

Dequantisation: `x_float = (code - zero) * scale`  where `zero = (1 << b) / 2`
(symmetric unsigned-to-signed mapping).

---

## 2. Legacy state (7-tuple)

`TurboQuantKCache.state` (property) returns a 7-tuple for backward compatibility
with checkpoints written before schema v2:

```python
(k_codes, k_scales, None, None, None, v_codes, v_scales)
```

Indices 2–4 (residual sign-sketch fields) are always `None` in the production
path.  `TurboQuantKCache.from_state(state, meta_state)` accepts this 7-tuple
plus a 18-element string tuple (`meta_state`) that encodes all config fields as
strings.

### meta_state tuple layout

```text
index  field
0      offset (str int)
1      d_head (str int)
2      d_pad  (str int or "")
3      v_dim  (str int or "")
4      v_pad  (str int or "")
5      dtype name (str or "")
6      main_bits
7      group_size
8      rotation
9      return_mode
10     scale_dtype
11     resid_scale_bits
12     v_bits
13     v_group_size
14     v_scale_dtype
15     v_enabled ("1" | "0")
16     block_tokens
17     state_version (str "2")
```

---

## 3. State validation

`validate_state(state, config=None)` in `turboquant/runtime/state.py` raises
`ValueError` for:

- Missing `schema_version`
- `schema_version` ≠ `STATE_SCHEMA_VERSION` (currently 2)
- Missing required scalar keys
- `k_packed.shape[2]` < `state["offset"]`
- Group count in `k_scales` inconsistent with `config.k_group_size` (if config provided)

---

## 4. PolarQuantPayload format

When `TurboQuantConfig(quantizer_mode="polar")` is used, `EncodedKeyBlock.packed_main`
and `EncodedKeyBlock.scales` are both `None`.  The compressed data lives in
`EncodedKeyBlock.polar`, a `PolarQuantPayload` dataclass:

| field | type | description |
|---|---|---|
| `angle_codes` | `list[mx.array]` (n_levels items) | per-level uint8 codebook indices; `angle_codes[ℓ]` has shape `[..., d // 2^{ℓ+1}]` |
| `final_radii` | `mx.array` float16 | `[..., d // 2^n_levels]` — the final Euclidean norms after all polar steps |
| `d_orig` | `int` | original head dimension before padding |
| `d_pad` | `int` | padded dimension (multiple of `2^n_levels`) |
| `n_levels` | `int` | recursion depth (default 4) |

### Codebook sizes

| level | bits | centroids | angle range | distribution |
|---|---|---|---|---|
| 1 | 4 | 16 | [0, 2π) | uniform (Gaussian preconditioning) |
| 2 | 2 | 4 | [0, π/2) | f(ψ) ∝ sin(2ψ) |
| 3 | 2 | 4 | [0, π/2) | f(ψ) ∝ sin²(2ψ) |
| 4 | 2 | 4 | [0, π/2) | f(ψ) ∝ sin⁴(2ψ) |

Codebooks are built offline via 1-D Lloyd's algorithm on 400k samples from each
distribution and cached globally in `turboquant.core.polar_quant._CB_CACHE`.

### Memory layout (d = 128, L = 4)

```text
angle_codes[0]: [B, H·T, 64]   uint8  →  64 × 4 bits  = 256 bits
angle_codes[1]: [B, H·T, 32]   uint8  →  32 × 2 bits  =  64 bits
angle_codes[2]: [B, H·T, 16]   uint8  →  16 × 2 bits  =  32 bits
angle_codes[3]: [B, H·T,  8]   uint8  →   8 × 2 bits  =  16 bits
final_radii:    [B, H·T,  8]   fp16   →   8 × 16 bits = 128 bits
                                          TOTAL = 496 bits / 128 dims = 3.875 bits/dim
```

No scale factors are stored — **zero memory overhead** compared to scalar quantisation.

### Residual interaction

PolarQuant does not use residual correction.  When `quantizer_mode="polar"`,
`encode_k_block` stores `ResidualPayload(mode="none", data={})` and the decode
path skips the residual step entirely.  To use QJL residuals, switch to
`quantizer_mode="scalar"`.

---

## 5. Convenience API (`TurboQuantKVCache`)

`TurboQuantKVCache` can be used directly as a benchmark stand-in or MLX-LM
adapter without going through `TurboQuantKCache`. Additional properties:

| property / method | description |
|---|---|
| `nbytes` | total compressed bytes (alias for `byte_size()`) |
| `k_packed` | `packed_main` tensor of the first block, or `None` if empty |
| `v_cache` | list of dense value tensors appended via `update_and_fetch` |
| `update_and_fetch(k, v)` | appends keys (compressed) and values (dense), returns `(TurboQuantKeysView, v)` |
| `memory_breakdown()` | `{"k_packed_main": int, "k_scales": int, "v_dense": int, "total": int}` |

`quantize_main` and `dequantize_main` are optional constructor arguments. When
omitted, a `GroupScalarQuantizer` is created automatically from
`config.k_bits` and `config.k_group_size`.

```python
from turboquant.runtime.kv_interface import TurboQuantKVCache
from turboquant.config import TurboQuantConfig
import mlx.core as mx

cfg = TurboQuantConfig(k_bits=3, k_group_size=64)
tq = TurboQuantKVCache(cfg)          # quantizers created automatically

k = mx.random.normal([1, 8, 256, 64], dtype=mx.float16)
v = mx.random.normal([1, 8, 256, 64], dtype=mx.float16)
view, _ = tq.update_and_fetch(k, v)

print(tq.nbytes)           # compressed byte count
print(tq.memory_breakdown())
# {'k_packed_main': 229376, 'k_scales': 16384, 'v_dense': 1048576, 'total': 1294336}
```

---

## 5. Serialisation example

```python
import mlx.core as mx
from turboquant.runtime.kv_interface import TurboQuantKVCache
from turboquant.config import TurboQuantConfig

cfg = TurboQuantConfig(k_bits=3, k_group_size=64)
tq = TurboQuantKVCache(cfg)
k = mx.random.normal([1, 8, 32, 64], dtype=mx.float16)
v = mx.random.normal([1, 8, 32, 64], dtype=mx.float16)
tq.update_and_fetch(k, v)

# Save
state = tq.state()
mx.savez("kv_state.npz", **{k: v for k, v in state.items()
                              if isinstance(v, mx.array)})

# Restore
restored = TurboQuantKVCache.from_state(
    state,
    quantize_main=tq.quantize_main,
    dequantize_main=tq.dequantize_main,
)
assert restored._offset == tq._offset
```
