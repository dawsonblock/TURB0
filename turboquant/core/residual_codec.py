from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import mlx.core as mx

from turboquant.config import TurboQuantConfig

from .qjl import QJLProjector
from .residual import decode_topk_residual, encode_topk_residual

ResidualMode = Literal["none", "topk", "qjl"]


@dataclass(slots=True)
class ResidualPayload:
    mode: ResidualMode
    data: dict[str, Any]


@runtime_checkable
class ResidualCodec(Protocol):
    mode: ResidualMode

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload: ...

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array | None: ...

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        raise NotImplementedError


class NoResidualCodec:
    mode: ResidualMode = "none"

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        return ResidualPayload(mode="none", data={})

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array | None:
        return None

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        batch_shape = q.shape[:-1]
        return mx.zeros(batch_shape, dtype=q.dtype)


class TopKResidualCodec:
    mode: ResidualMode = "topk"

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        vals, idx = encode_topk_residual(
            residual,
            k=config.residual_topk,
            group_size=config.k_group_size,
        )
        return ResidualPayload(
            mode="topk",
            data={
                "vals": vals,
                "idx": idx,
                "shape": tuple(residual.shape),
            },
        )

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        return decode_topk_residual(
            payload.data["vals"],
            payload.data["idx"],
            group_size=config.k_group_size,
        )

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        resid = self.decode(payload, config=config)
        return (q * resid).sum(axis=-1)


class QJLResidualCodec:
    mode: ResidualMode = "qjl"

    def __init__(self, *, proj_dim: int, seed: int):
        self.projector = QJLProjector(proj_dim=proj_dim, seed=seed)

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        bits, norms, meta = self.projector.encode(residual)
        return ResidualPayload(
            mode="qjl",
            data={
                "bits": bits,
                "norms": norms,
                "meta": meta.to_dict(),
            },
        )

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        return self.projector.decode(
            payload.data["bits"],
            payload.data["norms"],
            payload.data["meta"],
        )

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        return self.projector.dot_estimate(
            q,
            payload.data["bits"],
            payload.data["norms"],
            payload.data["meta"],
        )


def mode_contract(config: TurboQuantConfig) -> None:
    """Assert that algorithm ↔ residual-codec pairing is valid.

    Raises ``ValueError`` if a prod-mode config would produce no residual
    codec, or if an mse-mode config has a non-none residual mode.
    """
    if config.is_prod_mode() and config.residual_mode == "none":
        raise ValueError(
            "mode_contract: paper_prod_qjl requires a residual codec; "
            "residual_mode='none' is incompatible.  "
            "Use algorithm='paper_mse' for MSE-only quantization."
        )
    if config.is_mse_mode() and config.residual_mode != "none":
        raise ValueError(
            f"mode_contract: paper_mse requires residual_mode='none', "
            f"got residual_mode={config.residual_mode!r}."
        )


def build_residual_codec(config: TurboQuantConfig) -> ResidualCodec:
    """Build the residual codec for *config*.

    Dispatch order:
    1. Canonical algorithm family (``paper_mse`` / ``paper_prod_qjl``) takes
       precedence over ``residual_mode``.
    2. Legacy / experimental modes fall back to ``residual_mode``.
    """
    config.validate()

    if config.is_mse_mode():
        return NoResidualCodec()

    if config.is_prod_mode():
        return QJLResidualCodec(
            proj_dim=config.qjl_proj_dim,
            seed=config.qjl_seed,
        )

    if config.residual_mode == "none":
        return NoResidualCodec()

    if config.residual_mode == "topk":
        return TopKResidualCodec()

    if config.residual_mode == "qjl":
        return QJLResidualCodec(
            proj_dim=config.qjl_proj_dim,
            seed=config.qjl_seed,
        )

    raise ValueError(f"Unknown residual_mode: {config.residual_mode!r}")
