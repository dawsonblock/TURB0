# TurboQuant Cache Format

> **Schema version**: 4
> **Defined in**: `turboquant/runtime/state.py`

## 1. Canonical state dict (v4)

`TurboQuantKVCache.state()` emits a block-list payload. The top level keeps
scalar metadata, restore-time config checks, and algorithm identity, while the
compressed key payload lives under `blocks`.

| key | type | description |
| --- | --- | --- |
| `schema_version` | `int` | always `4` for the current format |
| `offset` | `int` | total number of cached tokens |
| `d_head` | `int` | original head dimension |
| `d_pad` | `int` | padded head dimension used by the encoder |
| `v_dim` | `int` | original value dimension (reserved; `0` in the current key-only payload) |
| `v_pad` | `int` | padded value dimension (reserved; `0` in the current key-only payload) |
| `k_bits` ... `eps` | scalar config fields | stored restore-time config contract |
| `algorithm` | `str` | one of `paper_mse`, `paper_prod_qjl`, `legacy_topk`, `polarquant_exp` |
| `rotation_type` | `str` | rotation/preconditioning family used by the encoder |
| `residual_kind` | `str` | one of `none`, `qjl`, `topk` |
| `qjl_dim` | `int` | QJL projection dimension for QJL residual modes |
| `qjl_seed` | `int` | seed used to derive the QJL transform |
| `codebook_id` | `str` | serialized identifier for the scalar or polar codebook family |
| `main_bits` | `int` | effective main quantizer bit width for the key path |
| `residual_bits` | `int` | effective residual bit width (`1` for QJL, `0` otherwise) |
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
| `polar_payload` | `dict \| None` | present only for `polarquant_exp`; contains encoded angle codes and final radii |

When `algorithm == "polarquant_exp"`, `packed_main` and `scales` are `None` and `polar_payload` carries:

| key | type | description |
| --- | --- | --- |
| `angle_codes` | `list[str]` | base64-encoded per-level angle-code tensors |
| `final_radii` | `str` | base64-encoded final-radii tensor |
| `d_orig` | `int` | original head dimension before padding |
| `d_pad` | `int` | padded dimension used by PolarQuant |
| `n_levels` | `int` | recursive PolarQuant depth |

## 2. Legacy payloads

`validate_state()` still accepts older payloads used by legacy tests, fixtures,
and migration helpers.

- **v1** stores packed tensors directly at the top level.
- **v2** keeps the flat tensor payload and adds the stored config keys used by
    `_expect_config_match()`.
- **v3** moves to the block-list payload but does not persist the explicit
    algorithm metadata introduced in v4.

The current canonical format for new persistence is the v4 block list.

## 3. Legacy adapter compatibility

Some older helpers still traffic in legacy adapter state:

- `TurboQuantKCache.state` may still expose adapter-oriented compatibility state.
- `TurboQuantKVCache.from_state(...)` continues to accept the older nested
    payload `{blocks: [...], config: {...}}`.

Those compatibility paths remain for migration and eval helpers. The supported
public persistence contract for new state snapshots is
`TurboQuantKVCache.state()` with `schema_version == 4`.

## 4. Validation guarantees

`validate_state(state, config=None)` in `turboquant/runtime/state.py` enforces:

- required top-level scalar keys are present
- `schema_version` is one of `1`, `2`, `3`, or `4`
- v3/v4 payloads contain a real `blocks` list with serialized block dicts
- v4 payloads carry explicit algorithm metadata and reject malformed paper-mode states
- each serialized block has the required keys and primitive metadata types
- stored config fields match the live `TurboQuantConfig` when `config` is supplied
- legacy flat payloads still satisfy their tensor-shape checks when `config` is supplied

Validation is intentionally strict. Restore should fail loudly on state/config
drift or malformed payloads rather than silently loading a cache with divergent
decode behavior.
