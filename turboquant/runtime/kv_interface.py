from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import (
    EncodedKeyBlock,
    decode_k_block,
    encode_k_block,
)


@dataclass(slots=True)
class KVCacheState:
    blocks: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "config": self.config,
        }


class TurboQuantKVCache:
    """
    Encoded K-block store with state roundtrip and MLX adapter support.

    Stores one EncodedKeyBlock per appended chunk.
    """

    def __init__(
        self,
        *,
        config: TurboQuantConfig,
        quantize_main,
        dequantize_main,
    ):
        config.validate()
        self.config = config
        self.quantize_main = quantize_main
        self.dequantize_main = dequantize_main
        self._blocks: list[EncodedKeyBlock] = []

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()

    def append_keys(self, k: mx.array) -> EncodedKeyBlock:
        """
        Encode and append one key block.

        Expected input shape:
            [..., seq, d_head] or [seq, d_head]
        depending on caller convention.
        """
        block = encode_k_block(
            k,
            config=self.config,
            quantize_main=self.quantize_main,
            dequantize_main=self.dequantize_main,
        )
        self._blocks.append(block)
        return block

    def append_encoded_block(self, block: EncodedKeyBlock) -> None:
        self._blocks.append(block)

    def block(self, index: int) -> EncodedKeyBlock:
        return self._blocks[index]

    def iter_blocks(self):
        yield from self._blocks

    def decode_block_full(self, index: int) -> mx.array:
        return decode_k_block(
            self._blocks[index],
            config=self.config,
            dequantize_main=self.dequantize_main,
        )

    def byte_size(self):
        return sum(b.packed_main.nbytes + b.scales.nbytes for b in self._blocks)

    def state(self) -> KVCacheState:
        return KVCacheState(
            blocks=[b.to_dict() for b in self._blocks],
            config={
                "k_bits": self.config.k_bits,
                "k_group_size": self.config.k_group_size,
                "rotation": self.config.rotation,
                "rotation_pad_to_pow2": self.config.rotation_pad_to_pow2,
                "residual_mode": self.config.residual_mode,
                "residual_topk": self.config.residual_topk,
                "resid_scale_bits": self.config.resid_scale_bits,
                "qjl_proj_dim": self.config.qjl_proj_dim,
                "qjl_seed": self.config.qjl_seed,
                "qjl_bits": self.config.qjl_bits,
                "paper_faithful_mode": self.config.paper_faithful_mode,
                "return_mode": self.config.return_mode,
            },
        )

    @classmethod
    def from_state(
        cls,
        state: KVCacheState | dict[str, Any],
        *,
        quantize_main,
        dequantize_main,
    ) -> TurboQuantKVCache:
        if isinstance(state, dict):
            raw = state
        else:
            raw = state.to_dict()

        config = TurboQuantConfig(
            k_bits=int(raw["config"]["k_bits"]),
            k_group_size=int(raw["config"]["k_group_size"]),
            rotation=raw["config"]["rotation"],
            rotation_pad_to_pow2=bool(raw["config"]["rotation_pad_to_pow2"]),
            residual_mode=raw["config"]["residual_mode"],
            residual_topk=int(raw["config"]["residual_topk"]),
            resid_scale_bits=int(raw["config"]["resid_scale_bits"]),
            qjl_proj_dim=int(raw["config"]["qjl_proj_dim"]),
            qjl_seed=int(raw["config"]["qjl_seed"]),
            qjl_bits=int(raw["config"]["qjl_bits"]),
            paper_faithful_mode=bool(raw["config"]["paper_faithful_mode"]),
            return_mode=raw["config"]["return_mode"],
        )
        config.validate()

        cache = cls(
            config=config,
            quantize_main=quantize_main,
            dequantize_main=dequantize_main,
        )
        cache._blocks = [EncodedKeyBlock.from_dict(b) for b in raw["blocks"]]
        return cache
# Shim for mlx_lm compatibility

class TurboQuantKeysView:
    def __init__(self, cache, start: int, end: int):
        self.cache = cache
        self.start = start
        self.end = end

