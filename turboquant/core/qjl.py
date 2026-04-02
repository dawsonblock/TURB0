from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx


@dataclass(slots=True)
class QJLMeta:
    input_dim: int
    proj_dim: int
    seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QJLMeta:
        return cls(
            input_dim=int(data["input_dim"]),
            proj_dim=int(data["proj_dim"]),
            seed=int(data["seed"]),
        )


def _ensure_float(x: mx.array) -> mx.array:
    if x.dtype in (mx.float16, mx.bfloat16, mx.float32):
        return x
    return x.astype(mx.float32)


def pack_sign_bits(signs: mx.array) -> mx.array:
    """
    Minimal placeholder packer.

    Stores 0/1 as uint8 for correctness-first wiring.
    Replace with real bit packing later.
    """
    return signs.astype(mx.uint8)


def unpack_sign_bits(bits: mx.array) -> mx.array:
    """
    Map uint8 {0,1} -> float {-1,+1}
    """
    return bits.astype(mx.float32) * 2.0 - 1.0


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
        Crude proxy reconstruction for debug/fallback paths.

        This is not the final paper-grade residual recovery procedure.
        It only gives the runtime something shaped correctly while
        the direct dot-estimator path is integrated.
        """
        if isinstance(meta, dict):
            meta = QJLMeta.from_dict(meta)

        proj = self._projection(meta.input_dim)
        signed = unpack_sign_bits(bits)

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
            [..., k_len, proj_dim]
        norms:
            [..., k_len, 1]

        returns:
            [..., q_len, k_len]
        """
        if isinstance(meta, dict):
            meta = QJLMeta.from_dict(meta)

        q = _ensure_float(q)
        proj = self._projection(meta.input_dim)

        q_proj = q @ proj                      # [..., q_len, proj_dim]
        signed = unpack_sign_bits(bits)        # [..., k_len, proj_dim]


        if q_proj.shape[-3] != signed.shape[-3]:
            n_rep = q_proj.shape[-3] // signed.shape[-3]
            signed = mx.repeat(signed, n_rep, axis=-3)

        if q_proj.shape[-3] != signed.shape[-3]:
            n_rep = q_proj.shape[-3] // signed.shape[-3]
            signed = mx.repeat(signed, n_rep, axis=-3)
        scores = q_proj @ mx.swapaxes(signed, -1, -2)   # [..., q_len, k_len]

        norm_scale = norms.squeeze(-1)                  # [..., k_len]
        q_norm = mx.linalg.norm(q, axis=-1)             # [..., q_len]
        q_norm = mx.maximum(q_norm, mx.array(1e-8, dtype=q_norm.dtype))


        if q_norm.shape[-2] != norm_scale.shape[-2]:
            n_rep = q_norm.shape[-2] // norm_scale.shape[-2]
            norm_scale = mx.repeat(norm_scale, n_rep, axis=-2)


        if q_norm.shape[-2] != norm_scale.shape[-2]:
            n_rep = q_norm.shape[-2] // norm_scale.shape[-2]
            norm_scale = mx.repeat(norm_scale, n_rep, axis=-2)

        return scores * (norm_scale[..., None, :] / q_norm[..., :, None])


