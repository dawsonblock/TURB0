from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from turboquant.config import TurboQuantConfig

from .residual_codec import ResidualPayload, build_residual_codec


@dataclass(slots=True)
class EncodedKeyBlock:
    packed_main: mx.array | None     # None when quantizer_mode='polar'
    scales: mx.array | None          # None when quantizer_mode='polar'
    residual: ResidualPayload
    d_head: int
    d_rot: int
    d_quant: int
    polar: object = None             # PolarQuantPayload | None
    algorithm: str = "turboquant_prod"  # "turboquant_mse" | "turboquant_prod"
    orig_dim: int = 0                # original head-dim before padding (0 = same as d_head)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (arrays stored as base64 numpy bytes)."""
        import base64, io
        def _arr_to_b64(arr):
            if arr is None:
                return None
            buf = io.BytesIO()
            np.save(buf, np.array(arr))
            return base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "packed_main": _arr_to_b64(self.packed_main),
            "scales": _arr_to_b64(self.scales),
            "residual_mode": self.residual.mode,
            "residual_data_keys": list(self.residual.data.keys()),
            "d_head": self.d_head,
            "d_rot": self.d_rot,
            "d_quant": self.d_quant,
            "algorithm": self.algorithm,
            "orig_dim": self.orig_dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EncodedKeyBlock":
        """Restore from a dict produced by to_dict()."""
        import base64, io
        def _b64_to_arr(b64):
            if b64 is None:
                return None
            raw = base64.b64decode(b64.encode("ascii"))
            arr = np.load(io.BytesIO(raw))
            return mx.array(arr)

        packed_main = _b64_to_arr(data.get("packed_main"))
        scales = _b64_to_arr(data.get("scales"))
        residual = ResidualPayload(mode=data.get("residual_mode", "none"), data={})
        return cls(
            packed_main=packed_main,
            scales=scales,
            residual=residual,
            d_head=int(data.get("d_head", 0)),
            d_rot=int(data.get("d_rot", 0)),
            d_quant=int(data.get("d_quant", 0)),
            algorithm=data.get("algorithm", "turboquant_prod"),
            orig_dim=int(data.get("orig_dim", 0)),
        )


class TurboQuantPipeline:
    """High-level encode/decode pipeline for a single layer."""
    def __init__(self, config: TurboQuantConfig, layer_id: int = 0):
        self.config = config
        self.layer_id = layer_id
        # In a real implementation, this would hold fitted quantizer state.
        # For now, it delegates to the functional encode/decode helpers.
        from .quantizer import GroupScalarQuantizer
        if config.quantizer_mode == "polar":
            from .polar_quant import PolarQuantizer
            self._k_quant = PolarQuantizer()
        else:
            self._k_quant = GroupScalarQuantizer(n_bits=config.k_bits, group_size=config.k_group_size)
        self._v_quant = GroupScalarQuantizer(n_bits=config.v_bits, group_size=config.v_group_size)

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
    k: mx.array,
    *,
    config: TurboQuantConfig,
    quantize_main,
    dequantize_main,
) -> EncodedKeyBlock:
    """
    Encode a dense key block.

    Applies the configured rotation internally (pipeline owns the rotation
    contract).  Callers pass un-rotated K in the model's original coordinate
    space.
    """
    config.validate()

    from .rotation import FixedRotation
    orig_dim = int(k.shape[-1])
    rotation = FixedRotation.from_config(config, orig_dim)
    k_rot = rotation.apply(k)

    d_head = orig_dim
    d_rot = orig_dim

    k_quant_in, d_quant = pad_last_dim(k_rot, config.k_group_size)

    packed_main, scales = quantize_main(k_quant_in, config=config)

    # PolarQuant path: scales is None, packed_main is a PolarQuantPayload
    if scales is None:
        return EncodedKeyBlock(
            packed_main=None,
            scales=None,
            residual=ResidualPayload(mode="none", data={}),
            d_head=d_head,
            d_rot=d_rot,
            d_quant=d_quant,
            polar=packed_main,
            algorithm=config.algorithm,
            orig_dim=orig_dim,
        )

    # Scalar quantisation path
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
        algorithm=config.algorithm,
        orig_dim=orig_dim,
    )


def decode_k_block(
    block: EncodedKeyBlock,
    *,
    config: TurboQuantConfig,
    dequantize_main,
) -> mx.array:
    config.validate()

    from .rotation import FixedRotation
    orig_dim = block.orig_dim if block.orig_dim > 0 else block.d_head

    # PolarQuant path: bypass scalar dequantisation and residual correction
    if block.polar is not None:
        x_hat = dequantize_main(block.polar, None, config=config)
        x_trimmed = x_hat[..., : block.d_rot]
        rotation = FixedRotation.from_config(config, orig_dim)
        return rotation.invert(x_trimmed)

    main_hat = dequantize_main(block.packed_main, block.scales, config=config)

    codec = build_residual_codec(config)
    resid_hat = codec.decode(block.residual, config=config)

    if resid_hat is None:
        k_quant_hat = main_hat
    else:
        k_quant_hat = main_hat + resid_hat

    k_rot_hat = k_quant_hat[..., : block.d_rot]
    rotation = FixedRotation.from_config(config, orig_dim)
    return rotation.invert(k_rot_hat)
