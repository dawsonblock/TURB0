from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from turboquant.config import TurboQuantConfig
from turboquant.runtime.state import STATE_SCHEMA_VERSION, validate_state

if TYPE_CHECKING:
    from turboquant.core.pipeline import EncodedKeyBlock


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
        _q: Any
        if quantize_main is None or dequantize_main is None:
            if config.is_polar_mode():
                from turboquant.core.polar_quant import PolarQuantizer

                _q = PolarQuantizer()
            elif config.is_mse_mode() or config.is_prod_mode():
                from turboquant.core.quantizer import LloydMaxScalarQuantizer

                _q = LloydMaxScalarQuantizer(
                    n_bits=config.k_bits, group_size=config.k_group_size
                )
            else:
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
        # V-block storage — "paper_kv" mode encodes V alongside K.
        # "k_only" legacy mode stores dense values in v_cache.
        self.storage_mode: str = (
            "paper_kv" if (config.is_mse_mode() or config.is_prod_mode()) else "k_only"
        )
        self.v_blocks: list[Any] = []
        self.v_cache: list = []  # kept for backward compat + k_only mode
        self._v_quantize: Any | None
        self._v_dequantize: Any | None
        self._v_config: TurboQuantConfig | None

        if self.storage_mode == "paper_kv":
            from turboquant.core.quantizer import LloydMaxScalarQuantizer

            _v = LloydMaxScalarQuantizer(
                n_bits=config.v_bits, group_size=config.v_group_size
            )
            self._v_quantize = _v.quantize
            self._v_dequantize = _v.dequantize
            self._v_config = TurboQuantConfig.paper_mse(
                k_bits=config.v_bits,
                k_group_size=config.v_group_size,
                rotation=config.rotation,
                rotation_seed=config.rotation_seed,
            )
        else:
            self._v_quantize = None
            self._v_dequantize = None
            self._v_config = None

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()
        self.v_blocks.clear()
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
        return cast("EncodedKeyBlock", self._blocks[index])

    def iter_blocks(self):
        yield from self._blocks

    def decode_block_full(self, index: int):
        from turboquant.core.pipeline import decode_k_block

        return decode_k_block(
            self._blocks[index],
            config=self.config,
            dequantize_main=self.dequantize_main,
        )

    def decode_v_block(self, index: int):
        """Decode the *index*-th V block back to original coordinate space."""
        from turboquant.core.pipeline import decode_k_block

        assert self._v_config is not None
        assert self._v_dequantize is not None
        return decode_k_block(
            self.v_blocks[index],
            config=self._v_config,
            dequantize_main=self._v_dequantize,
        )

    def byte_size(self) -> int:
        def _residual_bytes(b):
            bits_arr = b.residual.data.get("bits", None)
            norms_arr = b.residual.data.get("norms", None)
            return (bits_arr.nbytes if hasattr(bits_arr, "nbytes") else 0) + (
                norms_arr.nbytes if hasattr(norms_arr, "nbytes") else 0
            )

        def _polar_bytes(b):
            if getattr(b, "polar", None) is None:
                return 0
            return b.polar.byte_size()

        k_bytes = sum(
            (b.packed_main.nbytes if b.packed_main is not None else 0)
            + (b.scales.nbytes if b.scales is not None else 0)
            + _residual_bytes(b)
            + _polar_bytes(b)
            for b in self._blocks
        )
        if self.storage_mode == "paper_kv":
            v_bytes = sum(
                (b.packed_main.nbytes if b.packed_main is not None else 0)
                + (b.scales.nbytes if b.scales is not None else 0)
                for b in self.v_blocks
            )
        else:
            v_bytes = sum(v.nbytes for v in self.v_cache if hasattr(v, "nbytes"))
        return int(k_bytes + v_bytes)

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
        """Append compressed keys and values.

        Returns ``(keys_view, values)``.

        In paper mode (``storage_mode == "paper_kv"``): encodes V blocks for
        memory-efficient storage.  Dense *values* are still returned so that
        callers can pass them directly to attention.

        This is the MLX-LM cache-adapter protocol convenience method, so
        TurboQuantKVCache can be used directly as a drop-in benchmark stand-in
        without going through TurboQuantKCache.
        """
        start = self._offset
        self.append_keys(keys)

        if self.storage_mode == "paper_kv":
            from turboquant.core.pipeline import encode_k_block

            assert self._v_config is not None
            assert self._v_quantize is not None
            assert self._v_dequantize is not None
            v_block = encode_k_block(
                values,
                config=self._v_config,
                quantize_main=self._v_quantize,
                dequantize_main=self._v_dequantize,
            )
            self.v_blocks.append(v_block)
            # v_cache stays empty in paper mode.
            # Attention uses decode_v_block().
        else:
            self.v_cache.append(values)

        return TurboQuantKeysView(self, start, self._offset), values

    def memory_breakdown(self) -> dict:
        """Return a mapping of buffer name → byte size."""
        k_main = sum(
            b.packed_main.nbytes if b.packed_main is not None else 0
            for b in self._blocks
        )
        k_scales = sum(
            b.scales.nbytes if b.scales is not None else 0 for b in self._blocks
        )
        k_polar = sum(
            b.polar.byte_size() if getattr(b, "polar", None) is not None else 0
            for b in self._blocks
        )
        k_residual = sum(
            (
                b.residual.data.get("bits", None).nbytes
                if hasattr(b.residual.data.get("bits", None), "nbytes")
                else 0
            )
            + (
                b.residual.data.get("norms", None).nbytes
                if hasattr(b.residual.data.get("norms", None), "nbytes")
                else 0
            )
            for b in self._blocks
        )

        if self.storage_mode == "paper_kv":
            v_main = sum(
                b.packed_main.nbytes if b.packed_main is not None else 0
                for b in self.v_blocks
            )
            v_scales = sum(
                b.scales.nbytes if b.scales is not None else 0 for b in self.v_blocks
            )
        else:
            v_main = sum(int(v.nbytes) for v in self.v_cache if hasattr(v, "nbytes"))
            v_scales = 0
            # Legacy dense path — values remain stored densely in v_cache.
        total = k_main + k_scales + k_polar + k_residual + v_main + v_scales
        return {
            "k_main": k_main,
            "k_scales": k_scales,
            "k_polar": k_polar,
            "k_residual": k_residual,
            "v_main": v_main,
            "v_scales": v_scales,
            "total": total,
        }

    def state(self) -> dict[str, Any]:
        """Return the canonical v4 block-list state dict for serialisation.

        The returned dict is compatible with
        ``turboquant.runtime.state.validate_state``.
        """
        algo = self.config.algorithm_family()
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
            "algorithm": algo,
            "rotation_type": self.config.rotation,
            "residual_kind": self.config.residual_mode,
            "qjl_dim": (
                self.config.qjl_proj_dim if self.config.residual_mode == "qjl" else 0
            ),
            "qjl_seed": (
                self.config.qjl_seed if self.config.residual_mode == "qjl" else 0
            ),
            "codebook_id": self._codebook_id_for_state(),
            "main_bits": self._main_bits_for_state(),
            "residual_bits": self._residual_bits_for_state(),
            "blocks": [b.to_dict() for b in self._blocks],
        }

    def _main_bits_for_state(self) -> int:
        if self.config.is_prod_mode():
            return max(1, self.config.k_bits - 1)
        return self.config.k_bits

    def _residual_bits_for_state(self) -> int:
        if self.config.residual_mode == "qjl":
            return self.config.qjl_bits
        return 0

    def _codebook_id_for_state(self) -> str:
        algo = self.config.algorithm_family()
        if algo == "paper_mse":
            main_bits = self._main_bits_for_state()
            return f"lloydmax-k{main_bits}-{self.config.rotation}"
        if algo == "paper_prod_qjl":
            return (
                f"lloydmax-qjl-k{self._main_bits_for_state()}-"
                f"m{self.config.qjl_proj_dim}-{self.config.rotation}"
            )
        if algo == "polarquant_exp":
            return "polar-angle-codebook-exp"
        return f"legacy-topk-k{self.config.k_bits}-{self.config.rotation}"

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        quantize_main,
        dequantize_main,
    ) -> TurboQuantKVCache:
        """Reconstruct a cache from the current block-list state dict or the
        legacy nested format ``{blocks: [...], config: {...}}``.
        """
        from turboquant.core.pipeline import EncodedKeyBlock

        # Support legacy nested format: {blocks: [...], config: {...}}
        if "config" in state and isinstance(state.get("config"), dict):
            cfg = state["config"]
            config = TurboQuantConfig.from_legacy_kwargs(**cfg)
            blocks_data = state.get("blocks", [])
            offset = state.get("offset", 0)
            d_head = state.get("d_head", 0)
            d_pad = state.get("d_pad", 0)
        else:
            # Current block-list format produced by state()
            algorithm = state.get("algorithm", "paper_prod_qjl")
            config = TurboQuantConfig(
                k_bits=int(state.get("k_bits", 3)),
                k_group_size=int(state.get("k_group_size", 64)),
                v_bits=int(state.get("v_bits", 4)),
                v_group_size=int(state.get("v_group_size", 64)),
                v_enabled=bool(state.get("v_enabled", True)),
                rotation=state.get("rotation", "hadamard"),
                rotation_seed=int(state.get("rotation_seed", 1337)),
                residual_mode=state.get(
                    "residual_kind",
                    state.get("residual_mode", "qjl"),
                ),
                residual_topk=int(state.get("residual_topk", 0)),
                scale_dtype=state.get("scale_dtype", "float16"),
                v_scale_dtype=state.get("v_scale_dtype", "float16"),
                eps=float(state.get("eps", 1e-6)),
                qjl_proj_dim=int(state.get("qjl_dim", state.get("qjl_proj_dim", 64))),
                qjl_seed=int(state.get("qjl_seed", 42)),
                qjl_bits=int(state.get("residual_bits", state.get("qjl_bits", 1))),
                algorithm=algorithm,
                quantizer_mode=(
                    "polar"
                    if TurboQuantConfig.normalize_algorithm(algorithm)
                    == "polarquant_exp"
                    else "scalar"
                ),
            )
            validate_state(state, config)
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
