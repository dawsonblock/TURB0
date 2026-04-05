from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from turboquant.config import TurboQuantConfig


@dataclass(slots=True)
class QJLMeta:
    input_dim: int
    proj_dim: int
    seed: int
    algorithm: str = "paper_prod_qjl"

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "seed": self.seed,
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QJLMeta:
        return cls(
            input_dim=int(data["input_dim"]),
            proj_dim=int(data["proj_dim"]),
            seed=int(data["seed"]),
            algorithm=TurboQuantConfig.normalize_algorithm(
                str(data.get("algorithm", "paper_prod_qjl"))
            ),
        )


def _ensure_float(x: mx.array) -> mx.array:
    if x.dtype in (mx.float16, mx.bfloat16, mx.float32):
        return x
    return x.astype(mx.float32)


def pack_sign_bits(signs: mx.array) -> mx.array:
    """Pack binary {0,1} sign values [..., n] → [..., ceil(n/8)] uint8, LSB-first.

    Each byte stores 8 sign bits: bit-0 of byte-j holds sign j*8+0, bit-1
    holds sign j*8+1, and so on.  This achieves the paper's claimed 1-bit
    QJL residual storage exactly.
    """
    *prefix, n = signs.shape
    signs_u8 = signs.astype(mx.uint8)

    # Pad to a multiple of 8
    n_pad = ((n + 7) // 8) * 8
    if n_pad > n:
        pad = mx.zeros((*prefix, n_pad - n), dtype=mx.uint8)
        signs_u8 = mx.concatenate([signs_u8, pad], axis=-1)

    n_bytes = n_pad // 8
    # Reshape to groups of 8 bits and pack LSB-first into one byte each
    grouped = signs_u8.reshape(*prefix, n_bytes, 8).astype(mx.uint32)
    shifts = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint32)
    packed = mx.sum(mx.left_shift(grouped, shifts), axis=-1).astype(mx.uint8)
    return packed  # shape [..., ceil(n/8)]


def unpack_sign_bits(packed: mx.array, n: int) -> mx.array:
    """Unpack [..., ceil(n/8)] uint8 packed bits → float32 {-1, +1} [..., n].

    Inverse of :func:`pack_sign_bits`.  ``n`` must equal the original
    number of sign bits before packing.
    """
    *prefix, n_bytes = packed.shape
    shifts = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint32)
    # Broadcast: [..., n_bytes, 1] >> [8] → [..., n_bytes, 8]
    expanded = mx.right_shift(
        packed.reshape(*prefix, n_bytes, 1).astype(mx.uint32),
        shifts,
    )
    bits = mx.bitwise_and(expanded, mx.array(1, dtype=mx.uint32))
    # Flatten and trim padding
    flat = bits.reshape(*prefix, n_bytes * 8)[..., :n]
    return flat.astype(mx.float32) * 2.0 - 1.0


class QJLProjector:
    def __init__(self, *, proj_dim: int, seed: int):
        if proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {proj_dim}")
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self._proj_cache: dict[tuple[int, int, int], mx.array] = {}

    def _projection(self, input_dim: int) -> mx.array:
        key = (int(input_dim), self.proj_dim, self.seed)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached

        rng = mx.random.key(self.seed)
        proj = mx.random.normal(
            shape=(input_dim, self.proj_dim),
            key=rng,
        ).astype(mx.float32)

        proj = proj / mx.sqrt(mx.array(float(self.proj_dim), dtype=mx.float32))
        self._proj_cache[key] = proj
        return proj

    def encode(self, residual: mx.array) -> tuple[mx.array, mx.array, QJLMeta]:
        residual = _ensure_float(residual)
        input_dim = int(residual.shape[-1])

        proj = self._projection(input_dim)
        sketch = residual @ proj

        norms = mx.linalg.norm(residual, axis=-1, keepdims=True)
        bits = pack_sign_bits(sketch >= 0)

        meta = QJLMeta(
            input_dim=input_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
        )
        return bits, norms, meta

    def decode(
        self,
        bits: mx.array,
        norms: mx.array,
        meta: QJLMeta | dict[str, Any],
    ) -> mx.array:
        """
        Proxy reconstruction for debug/fallback paths.

        The primary attention path uses dot_estimate() for inner-product
        estimation without full reconstruction.  This decode() is provided
        for diagnostic use and the TopK-residual fallback path.
        """
        if isinstance(meta, dict):
            meta = QJLMeta.from_dict(meta)

        proj = self._projection(meta.input_dim)
        signed = unpack_sign_bits(bits, meta.proj_dim)

        proxy = signed @ proj.T
        proxy_norm = mx.linalg.norm(proxy, axis=-1, keepdims=True)
        proxy_norm = mx.maximum(
            proxy_norm,
            mx.array(1e-8, dtype=proxy.dtype),
        )
        proxy = proxy * (norms / proxy_norm)
        return proxy

    def dot_estimate(
        self,
        q: mx.array,
        bits: mx.array,
        norms: mx.array,
        meta: QJLMeta | dict[str, Any],
    ) -> mx.array:
        """
        Estimate q · residual for all query/key pairs.

        q:
            [..., q_len, d]
        bits:
            [..., k_len, ceil(proj_dim/8)]  (1-bit packed, LSB-first uint8)
        norms:
            [..., k_len, 1]

        returns:
            [..., q_len, k_len]
        """
        if isinstance(meta, dict):
            meta = QJLMeta.from_dict(meta)

        q = _ensure_float(q)
        proj = self._projection(meta.input_dim)

        q_proj = q @ proj  # [..., q_len, proj_dim]
        signed = unpack_sign_bits(bits, meta.proj_dim)  # [..., k_len, proj_dim]

        if q_proj.shape[-3] != signed.shape[-3]:
            n_rep = q_proj.shape[-3] // signed.shape[-3]
            signed = mx.repeat(signed, n_rep, axis=-3)

        scores = q_proj @ mx.swapaxes(signed, -1, -2)  # [..., q_len, k_len]

        norm_scale = norms.squeeze(-1)  # [..., k_len]
        q_norm = mx.linalg.norm(q, axis=-1)  # [..., q_len]
        q_norm = mx.maximum(q_norm, mx.array(1e-8, dtype=q_norm.dtype))

        if q_norm.shape[-2] != norm_scale.shape[-2]:
            n_rep = q_norm.shape[-2] // norm_scale.shape[-2]
            norm_scale = mx.repeat(norm_scale, n_rep, axis=-2)

        return scores * (norm_scale[..., None, :] / q_norm[..., :, None])

    def estimate_inner_product(
        self,
        q: mx.array,
        bits: mx.array,
        norms: mx.array,
        meta: QJLMeta | dict[str, Any],
    ) -> mx.array:
        """Primary high-level API for inner-product estimation.

        Estimates ``q · residual`` for all query/key pairs in rotated space,
        without fully reconstructing the residual vector.

        Alias of ``dot_estimate``; prefer this name in new code.
        """
        return self.dot_estimate(q, bits, norms, meta)
