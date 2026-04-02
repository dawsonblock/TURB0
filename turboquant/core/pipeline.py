from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from turboquant.config import TurboQuantConfig

from .residual_codec import ResidualPayload, build_residual_codec


@dataclass(slots=True)
class EncodedKeyBlock:
    packed_main: mx.array
    scales: mx.array
    residual: ResidualPayload
    d_head: int
    d_rot: int
    d_quant: int


class TurboQuantPipeline:
    """High-level encode/decode pipeline for a single layer."""
    def __init__(self, config: TurboQuantConfig, layer_id: int = 0):
        self.config = config
        self.layer_id = layer_id
        # In a real implementation, this would hold fitted quantizer state.
        # For now, it delegates to the functional encode/decode helpers.
        from .quantizer import GroupScalarQuantizer
        self._k_quant = GroupScalarQuantizer(bits=config.k_bits, group_size=config.k_group_size)
        self._v_quant = GroupScalarQuantizer(bits=config.v_bits, group_size=config.v_group_size)

    def _get_k_quant(self): return self._k_quant
    def _get_v_quant(self): return self._v_quant

    def encode_k(self, k: mx.array) -> EncodedKeyBlock:
        return encode_k_block(
            k, 
            config=self.config, 
            quantize_main=self._k_quant.quantize,
            dequantize_main=self._k_quant.dequantize
        )

    def decode_k(self, block: EncodedKeyBlock) -> mx.array:
        return decode_k_block(
            block, 
            config=self.config, 
            dequantize_main=self._k_quant.dequantize
        )


def pad_last_dim(x: mx.array, multiple: int) -> tuple[mx.array, int]:
    d = int(x.shape[-1])
    d2 = ((d + multiple - 1) // multiple) * multiple
    if d2 == d:
        return x, d2

    pad = d2 - d
    zeros = mx.zeros((*x.shape[:-1], pad), dtype=x.dtype)
    return mx.concatenate([x, zeros], axis=-1), d2


def encode_k_block(
    k_rot: mx.array,
    *,
    config: TurboQuantConfig,
    quantize_main,
    dequantize_main,
) -> EncodedKeyBlock:
    """
    Transitional version:
    expects already-rotated K until rotation.py is patched.
    """
    config.validate()

    d_head = int(k_rot.shape[-1])
    d_rot = d_head

    k_quant_in, d_quant = pad_last_dim(k_rot, config.k_group_size)

    packed_main, scales = quantize_main(k_quant_in, config=config)
    main_hat = dequantize_main(packed_main, scales, config=config)

    residual = k_quant_in - main_hat
    codec = build_residual_codec(config)
    residual_payload = codec.encode(residual, config=config)

    return EncodedKeyBlock(
        packed_main=packed_main,
        scales=scales,
        residual=residual_payload,
        d_head=d_head,
        d_rot=d_rot,
        d_quant=d_quant,
    )


def decode_k_block(
    block: EncodedKeyBlock,
    *,
    config: TurboQuantConfig,
    dequantize_main,
) -> mx.array:
    config.validate()

    main_hat = dequantize_main(block.packed_main, block.scales, config=config)

    codec = build_residual_codec(config)
    resid_hat = codec.decode(block.residual, config=config)

    if resid_hat is None:
        k_quant_hat = main_hat
    else:
        k_quant_hat = main_hat + resid_hat

    return k_quant_hat[..., : block.d_rot]
