# TurboQuant Cache Format

> **Schema version**: 3
> **Defined in**: `turboquant/runtime/state.py`

## 1. Canonical state dict (v3)

`TurboQuantKVCache.state()` now emits a block-list payload. The top level keeps
the scalar metadata and restore-time config checks, while the compressed key
payload lives under `blocks`.

| key | type | description |
| --- | --- | --- |
| `schema_version` | `int` | always `3` for the current format |
| `offset` | `int` | total number of cached tokens |
| `d_head` | `int` | original head dimension |
| `d_pad` | `int` | padded head dimension used by the encoder |
| `v_dim` | `int` | original value dimension (reserved; `0` in the current key-only payload) |
| `v_pad` | `int` | padded value dimension (reserved; `0` in the current key-only payload) |
| `k_bits` ... `eps` | scalar config fields | stored restore-time config contract |
| `blocks` | `list[dict]` | serialized `EncodedKeyBlock` payloads in append order |

Each entry in `blocks` is produced by `EncodedKeyBlock.to_dict()` and contains:

| key | type | description |
| --- | --- | --- |
| `packed_main` | `str \| None` | base64-encoded main packed tensor, or `None` |
| `scales` | `str \| None` | base64-encoded scales tensor, or `None` |
| `residual_mode` | `str` | residual codec mode |
| `residual_data_keys` | `list[str]` | residual payload field names |
| `d_head` | `int` | original block head dimension |
| `d_rot` | `int` | rotated dimension |
| `d_quant` | `int` | quantized dimension |
| `algorithm` | `str` | encoder algorithm name |
| `orig_dim` | `int` | original dimension before padding |

## 2. Legacy flat payloads (v1 and v2)

`validate_state()` still accepts the older flat payloads used by legacy tests,
fixtures, and migration helpers.

- **v1** stores packed tensors directly at the top level (`k_packed`, `k_scales`,
    residual arrays, `v_packed`, `v_scales`).
- **v2** keeps the flat tensor payload and adds the stored config keys used by
    `_expect_config_match()` plus optional `k_calibrated_scales` and
    `v_calibrated_scales`.

Those older payloads remain structurally valid, but the current canonical format
for new persistence is the v3 block list.

## 3. Legacy adapter compatibility

Some older helpers still traffic in legacy adapter state. Two compatibility
surfaces remain documented:

- `TurboQuantKCache.state` can still expose the older tuple-oriented adapter state.
- `TurboQuantKVCache.from_state(...)` continues to accept the legacy nested
    adapter payload `{blocks: [...], config: {...}}`.

These compatibility paths exist for migration and eval helpers. The supported
public persistence contract for new state snapshots is `TurboQuantKVCache.state()`
with `schema_version == 3`.

## 4. Validation guarantees

`validate_state(state, config=None)` in `turboquant/runtime/state.py` enforces:

- required top-level scalar keys are present
- `schema_version` is one of `1`, `2`, or `3`
- v3 payloads contain a real `blocks` list with serialized block dicts
- each serialized block has the required keys and primitive metadata types
- stored config fields match the live `TurboQuantConfig` when `config` is supplied
- legacy flat payloads still satisfy their tensor-shape checks when `config` is supplied

Validation is intentionally strict. Restore should fail loudly on state/config
drift or malformed payloads rather than silently loading a cache with divergent
decode behavior.
