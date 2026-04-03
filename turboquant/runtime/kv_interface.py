from __future__ import annotations

from typing import Any

from turboquant.config import TurboQuantConfig
from turboquant.runtime.state import STATE_SCHEMA_VERSION


class TurboQuantKVCache:
    """
    Encoded K-block store with state roundtrip and MLX adapter support.

    Stores one EncodedKeyBlock per appended chunk.
    """

    def __init__(
        self,
        config: TurboQuantConfig | None = None,
        *,
        quantize_main=None,
        dequantize_main=None,
    ):
        if config is None:
            raise TypeError("TurboQuantKVCache requires a TurboQuantConfig.")
        config.validate()
        self.config = config
        if quantize_main is None or dequantize_main is None:
            from turboquant.core.quantizer import GroupScalarQuantizer
            _q = GroupScalarQuantizer(
                n_bits=config.k_bits, group_size=config.k_group_size
            )
            if quantize_main is None:
                quantize_main = _q.quantize
            if dequantize_main is None:
                dequantize_main = _q.dequantize
        self.quantize_main = quantize_main
        self.dequantize_main = dequantize_main
        self._blocks: list[Any] = []
        self._offset: int = 0
        self._d_head: int = 0
        self._d_pad: int = 0
        self.v_cache: list = []

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()
        self.v_cache.clear()
        self._offset = 0
        self._d_head = 0
        self._d_pad = 0

    def append_keys(self, k):
        """
        Encode one key block and append it to the store.

        Expected input shape:
            [..., seq, d_head] or [seq, d_head]
        depending on caller convention.

        Rotation is applied internally by encode_k_block (pipeline contract).
        """
        from turboquant.core.pipeline import encode_k_block

        block = encode_k_block(
            k,
            config=self.config,
            quantize_main=self.quantize_main,
            dequantize_main=self.dequantize_main,
        )
        self._blocks.append(block)
        # Track cumulative token offset and head dimensions.
        shape = k.shape
        if len(shape) >= 2:
            self._offset += shape[-2]
            self._d_head = shape[-1]
        return block

    def append_encoded_block(self, block: EncodedKeyBlock) -> None:
        self._blocks.append(block)

    def block(self, index: int) -> EncodedKeyBlock:
        return self._blocks[index]

    def iter_blocks(self):
        yield from self._blocks

    def decode_block_full(self, index: int):
        from turboquant.core.pipeline import decode_k_block

        return decode_k_block(
            self._blocks[index],
            config=self.config,
            dequantize_main=self.dequantize_main,
        )

    def byte_size(self):
        k_bytes = sum(
            (b.packed_main.nbytes if b.packed_main is not None else 0)
            + (b.scales.nbytes if b.scales is not None else 0)
            for b in self._blocks
        )
        v_bytes = sum(v.nbytes for v in self.v_cache if hasattr(v, "nbytes"))
        return k_bytes + v_bytes

    @property
    def nbytes(self) -> int:
        """Total compressed bytes (alias for byte_size())."""
        return self.byte_size()

    @property
    def k_packed(self):
        """Packed tensor of the first key block, or None if cache is empty."""
        if not self._blocks:
            return None
        return self._blocks[0].packed_main

    def update_and_fetch(self, keys, values):
        """Append *keys* (compressed) and *values* (dense), return (keys_view, values).

        This is the MLX-LM cache-adapter protocol convenience method, so
        TurboQuantKVCache can be used directly as a drop-in benchmark stand-in
        without going through TurboQuantKCache.
        """
        start = self._offset
        self.append_keys(keys)
        self.v_cache.append(values)
        return TurboQuantKeysView(self, start, self._offset), values

    def memory_breakdown(self) -> dict:
        """Return a mapping of buffer name → byte size."""
        k_main = sum(
            b.packed_main.nbytes if b.packed_main is not None else 0
            for b in self._blocks
        )
        k_scales = sum(
            b.scales.nbytes if b.scales is not None else 0
            for b in self._blocks
        )
        v_dense = sum(
            v.nbytes for v in self.v_cache if hasattr(v, "nbytes")
        )
        total = k_main + k_scales + v_dense
        return {
            "k_packed_main": k_main,
            "k_scales": k_scales,
            "v_dense": v_dense,
            "total": total,
        }

    def state(self) -> dict[str, Any]:
        """Return the canonical flat state dict for serialisation.

        The returned dict is compatible with
        ``turboquant.runtime.state.validate_state``.
        """
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "offset": self._offset,
            "d_head": self._d_head,
            "d_pad": self._d_pad,
            "v_dim": 0,
            "v_pad": 0,
            # V2 config fields
            "k_bits": self.config.k_bits,
            "k_group_size": self.config.k_group_size,
            "v_bits": self.config.v_bits,
            "v_group_size": self.config.v_group_size,
            "v_enabled": self.config.v_enabled,
            "rotation": self.config.rotation,
            "rotation_seed": self.config.rotation_seed,
            "residual_topk": self.config.residual_topk,
            "scale_dtype": self.config.scale_dtype,
            "v_scale_dtype": self.config.v_scale_dtype,
            "eps": self.config.eps,
            # Block data (extra key; not constrained by validate_state)
            "blocks": [b.to_dict() for b in self._blocks],
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        quantize_main,
        dequantize_main,
    ) -> TurboQuantKVCache:
        """Reconstruct a cache from a flat state dict (current format) or the
        legacy nested format ``{blocks: [...], config: {...}}``.
        """
        from turboquant.core.pipeline import EncodedKeyBlock

        # Support legacy nested format: {blocks: [...], config: {...}}
        if "config" in state and isinstance(state.get("config"), dict):
            cfg = state["config"]
            config = TurboQuantConfig(
                k_bits=int(cfg.get("k_bits", 3)),
                k_group_size=int(cfg.get("k_group_size", 64)),
                rotation=cfg.get("rotation", "hadamard"),
                rotation_pad_to_pow2=bool(cfg.get("rotation_pad_to_pow2", True)),
                residual_mode=cfg.get("residual_mode", "qjl"),
                residual_topk=int(cfg.get("residual_topk", 0)),
                resid_scale_bits=int(cfg.get("resid_scale_bits", 8)),
                qjl_proj_dim=int(cfg.get("qjl_proj_dim", 64)),
                qjl_seed=int(cfg.get("qjl_seed", 42)),
                qjl_bits=int(cfg.get("qjl_bits", 1)),
                algorithm=cfg.get("algorithm", "turboquant_prod"),
                return_mode=cfg.get("return_mode", "view"),
            )
            blocks_data = state.get("blocks", [])
            offset = state.get("offset", 0)
            d_head = state.get("d_head", 0)
            d_pad = state.get("d_pad", 0)
        else:
            # Current flat format produced by state()
            config = TurboQuantConfig(
                k_bits=int(state.get("k_bits", 3)),
                k_group_size=int(state.get("k_group_size", 64)),
                v_bits=int(state.get("v_bits", 4)),
                v_group_size=int(state.get("v_group_size", 64)),
                v_enabled=bool(state.get("v_enabled", True)),
                rotation=state.get("rotation", "hadamard"),
                rotation_seed=int(state.get("rotation_seed", 1337)),
                residual_topk=int(state.get("residual_topk", 0)),
                scale_dtype=state.get("scale_dtype", "float16"),
                v_scale_dtype=state.get("v_scale_dtype", "float16"),
                eps=float(state.get("eps", 1e-6)),
                algorithm=state.get("algorithm", "turboquant_prod"),
            )
            blocks_data = state.get("blocks", [])
            offset = int(state.get("offset", 0))
            d_head = int(state.get("d_head", 0))
            d_pad = int(state.get("d_pad", 0))

        config.validate()
        cache = cls(
            config=config,
            quantize_main=quantize_main,
            dequantize_main=dequantize_main,
        )
        cache._blocks = [EncodedKeyBlock.from_dict(b) for b in blocks_data]
        cache._offset = offset
        cache._d_head = d_head
        cache._d_pad = d_pad
        return cache
# Shim for mlx_lm compatibility

class TurboQuantKeysView:
    def __init__(self, cache, start: int, end: int):
        self.cache = cache
        self.start = start
        self.end = end

