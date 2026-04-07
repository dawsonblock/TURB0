from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

_CANONICAL_ALGORITHMS = frozenset(
    {"paper_mse", "paper_prod_qjl", "legacy_topk", "polarquant_exp"}
)
_LEGACY_ALGORITHM_ALIASES = {
    "turboquant_mse": "paper_mse",
    "turboquant_prod": "paper_prod_qjl",
    "paper_prod": "paper_prod_qjl",
}

_PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "paper_mse": {
        "factory": "paper_mse",
        "kwargs": {
            "k_bits": 3,
            "k_group_size": 64,
            "v_bits": 4,
            "v_group_size": 64,
            "rotation": "hadamard",
        },
        "classification": "paper-facing",
        "paper_facing": True,
        "canonical_preset": "paper_mse",
        "comparison_role": "scalar-only paper baseline",
        "notes": "Paper-facing MSE stage.",
        "algorithm_aliases": ("turboquant_mse",),
    },
    "paper_prod_qjl": {
        "factory": "paper_prod_qjl",
        "kwargs": {
            "k_bits": 3,
            "k_group_size": 64,
            "v_bits": 4,
            "v_group_size": 64,
            "rotation": "hadamard",
            "qjl_proj_dim": 64,
        },
        "classification": "paper-facing",
        "paper_facing": True,
        "canonical_preset": "paper_prod_qjl",
        "comparison_role": "full two-stage paper path",
        "notes": "Primary paper-facing two-stage preset using a 1-bit QJL residual.",
        "algorithm_aliases": ("turboquant_prod",),
    },
    "paper_prod": {
        "factory": "paper_prod_qjl",
        "kwargs": {
            "k_bits": 3,
            "k_group_size": 64,
            "v_bits": 4,
            "v_group_size": 64,
            "rotation": "hadamard",
            "qjl_proj_dim": 64,
        },
        "classification": "paper-facing",
        "paper_facing": True,
        "canonical_preset": "paper_prod_qjl",
        "comparison_role": "paper-facing alias for the full two-stage path",
        "notes": "Paper-facing production-style preset using a 1-bit QJL residual.",
        "algorithm_aliases": ("paper_prod_qjl", "turboquant_prod"),
    },
    "polarquant_exp": {
        "factory": "polarquant_exp",
        "kwargs": {
            "k_bits": 3,
            "k_group_size": 64,
            "v_bits": 4,
            "v_group_size": 64,
            "rotation": "random_orthogonal",
        },
        "classification": "supported non-paper-facing",
        "paper_facing": False,
        "canonical_preset": "polarquant_exp",
        "comparison_role": "closest available first-stage-only research path",
        "notes": "Supported non-paper-facing PolarQuant branch with family-scoped runtime and quality certification.",
        "algorithm_aliases": (),
    },
    "legacy_topk": {
        "factory": "legacy_topk",
        "kwargs": {
            "k_bits": 4,
            "k_group_size": 32,
            "v_bits": 4,
            "v_group_size": 32,
            "rotation": "hadamard",
            "residual_topk": 2,
        },
        "classification": "compatibility-only",
        "paper_facing": False,
        "canonical_preset": "legacy_topk",
        "comparison_role": "explicit legacy top-k compatibility surface",
        "notes": "Explicit legacy top-k compatibility preset; not part of the paper-facing contract.",
        "algorithm_aliases": (),
    },
    "balanced": {
        "factory": "legacy_topk",
        "kwargs": {
            "k_bits": 4,
            "k_group_size": 32,
            "v_bits": 4,
            "v_group_size": 32,
            "rotation": "hadamard",
            "residual_topk": 2,
        },
        "classification": "compatibility-only",
        "paper_facing": False,
        "canonical_preset": "balanced",
        "comparison_role": "legacy top-k comparison preset",
        "notes": "Legacy top-k compatibility preset; not part of the paper-facing contract.",
        "algorithm_aliases": (),
    },
    "max_quality": {
        "factory": "legacy_topk",
        "kwargs": {
            "k_bits": 4,
            "k_group_size": 16,
            "v_bits": 8,
            "v_group_size": 16,
            "rotation": "hadamard",
            "residual_topk": 4,
        },
        "classification": "compatibility-only",
        "paper_facing": False,
        "canonical_preset": "max_quality",
        "comparison_role": "higher-precision legacy top-k comparison preset",
        "notes": "Legacy top-k compatibility preset; not part of the paper-facing contract.",
        "algorithm_aliases": (),
    },
    "high_compression": {
        "factory": "paper_prod_qjl",
        "kwargs": {
            "k_bits": 3,
            "k_group_size": 64,
            "v_bits": 4,
            "v_group_size": 64,
            "rotation": "hadamard",
            "qjl_proj_dim": 64,
        },
        "classification": "compatibility-only",
        "paper_facing": False,
        "canonical_preset": "paper_prod_qjl",
        "comparison_role": "legacy alias for the two-stage paper path",
        "notes": "Legacy convenience alias for the QJL production-style preset.",
        "algorithm_aliases": ("paper_prod_qjl", "turboquant_prod"),
    },
}


@dataclass(slots=True)
class TurboQuantConfig:
    k_bits: int = 3
    k_group_size: int = 64
    v_bits: int = 4
    v_group_size: int = 64
    v_enabled: bool = True

    rotation: str = "hadamard"
    rotation_seed: int = 1337
    rotation_pad_to_pow2: bool = True

    residual_mode: str = "qjl"
    residual_topk: int = 0
    resid_scale_bits: int = 8

    scale_dtype: str = "float16"
    v_scale_dtype: str = "float16"
    eps: float = 1e-6
    # Reserved for backward compatibility with older configs and benchmarks.
    # The current streaming-attention hot path does not read this as a live
    # runtime control.
    block_tokens: int = 256

    qjl_proj_dim: int = 64
    qjl_seed: int = 42
    qjl_bits: int = 1

    quantizer_mode: str = "scalar"  # "scalar" | "polar"
    algorithm: str = "paper_prod_qjl"
    return_mode: str = "view"

    @staticmethod
    def normalize_algorithm(name: str) -> str:
        return _LEGACY_ALGORITHM_ALIASES.get(name, name)

    def algorithm_family(self) -> str:
        return self.normalize_algorithm(self.algorithm)

    def is_mse_mode(self) -> bool:
        """True when only scalar quantization is used (no QJL residual)."""
        return self.algorithm_family() == "paper_mse"

    def is_prod_mode(self) -> bool:
        """True when QJL residual is used for inner-product estimation."""
        return self.algorithm_family() == "paper_prod_qjl"

    def is_legacy_topk_mode(self) -> bool:
        return self.algorithm_family() == "legacy_topk"

    def is_polar_mode(self) -> bool:
        return self.algorithm_family() == "polarquant_exp"

    def effective_bits_per_channel_k(self, d: int) -> float:
        """Effective bits/channel for K cache at head-dim *d*."""
        b = self.k_bits
        g = self.k_group_size
        if self.is_prod_mode():
            p = self.qjl_proj_dim
            return (b - 1) + 16.0 / g + (p + 16.0) / d
        return b + 16.0 / g

    def effective_bits_per_channel_v(self, d: int) -> float:
        """Effective bits/channel for V cache at head-dim *d* (always MSE)."""
        return self.v_bits + 16.0 / self.v_group_size

    def effective_bits_per_channel_total(self, d: int) -> float:
        """Average of K and V effective bits/channel."""
        return (
            self.effective_bits_per_channel_k(d) + self.effective_bits_per_channel_v(d)
        ) / 2.0

    def validate(self) -> None:
        algo = self.algorithm_family()

        if algo not in _CANONICAL_ALGORITHMS:
            raise ValueError(
                "algorithm must be one of "
                f"{sorted(_CANONICAL_ALGORITHMS)}, got {self.algorithm!r}"
            )

        if self.k_bits <= 0 or self.k_bits > 8:
            raise ValueError(f"k_bits must be in [1, 8], got {self.k_bits}")

        if self.k_group_size <= 0:
            raise ValueError(f"k_group_size must be > 0, got {self.k_group_size}")

        if self.v_enabled:
            if self.v_bits <= 0 or self.v_bits > 8:
                raise ValueError(f"v_bits must be in [1, 8], got {self.v_bits}")
            if self.v_group_size <= 0:
                raise ValueError(f"v_group_size must be > 0, got {self.v_group_size}")

        if self.rotation not in {"hadamard", "identity", "random_orthogonal"}:
            raise ValueError(f"Unsupported rotation: {self.rotation}")

        if self.residual_mode not in {"none", "topk", "qjl"}:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        if self.quantizer_mode not in {"scalar", "polar"}:
            raise ValueError(
                f"Unsupported quantizer_mode: {self.quantizer_mode!r}; "
                "expected 'scalar' or 'polar'"
            )

        if algo == "paper_mse":
            if self.quantizer_mode != "scalar":
                raise ValueError("paper_mse requires quantizer_mode='scalar'")
            if self.residual_mode != "none":
                raise ValueError(
                    "paper_mse requires residual_mode='none', "
                    f"got {self.residual_mode!r}"
                )

        if algo == "paper_prod_qjl":
            if self.quantizer_mode != "scalar":
                raise ValueError("paper_prod_qjl requires quantizer_mode='scalar'")
            if self.residual_mode != "qjl":
                raise ValueError(
                    "paper_prod_qjl requires residual_mode='qjl', "
                    f"got {self.residual_mode!r}"
                )

        if algo == "legacy_topk":
            if self.quantizer_mode != "scalar":
                raise ValueError("legacy_topk requires quantizer_mode='scalar'")
            if self.residual_mode != "topk":
                raise ValueError(
                    "legacy_topk requires residual_mode='topk', "
                    f"got {self.residual_mode!r}"
                )

        if algo == "polarquant_exp":
            if self.quantizer_mode != "polar":
                raise ValueError("polarquant_exp requires quantizer_mode='polar'")
            if self.rotation == "identity":
                raise ValueError(
                    "polarquant_exp expects a randomized preconditioning rotation"
                )

        if self.residual_mode == "topk" and self.residual_topk <= 0:
            raise ValueError("residual_topk must be > 0 when residual_mode='topk'")

        if self.residual_mode == "qjl":
            if self.qjl_bits != 1:
                raise ValueError(
                    f"Only 1-bit QJL is currently supported, got {self.qjl_bits}"
                )
            if self.qjl_proj_dim <= 0:
                raise ValueError(f"qjl_proj_dim must be > 0, got {self.qjl_proj_dim}")

    @classmethod
    def paper_mse(cls, **kwargs) -> TurboQuantConfig:
        return cls(
            algorithm="paper_mse",
            quantizer_mode="scalar",
            residual_mode="none",
            **kwargs,
        )

    @classmethod
    def paper_prod_qjl(cls, **kwargs) -> TurboQuantConfig:
        return cls(
            algorithm="paper_prod_qjl",
            quantizer_mode="scalar",
            residual_mode="qjl",
            qjl_bits=1,
            **kwargs,
        )

    @classmethod
    def legacy_topk(cls, **kwargs) -> TurboQuantConfig:
        return cls(
            algorithm="legacy_topk",
            quantizer_mode="scalar",
            residual_mode="topk",
            residual_topk=max(int(kwargs.pop("residual_topk", 2)), 1),
            **kwargs,
        )

    @classmethod
    def polarquant_exp(cls, **kwargs) -> TurboQuantConfig:
        return cls(
            algorithm="polarquant_exp",
            quantizer_mode="polar",
            residual_mode=kwargs.pop("residual_mode", "none"),
            **kwargs,
        )

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        return tuple(_PRESET_REGISTRY.keys())

    @classmethod
    def preset_metadata(cls, name: str) -> dict[str, Any]:
        if name not in _PRESET_REGISTRY:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(_PRESET_REGISTRY.keys())}"
            )

        spec = _PRESET_REGISTRY[name]
        cfg = cls.from_preset(name)
        return {
            "name": name,
            "factory": spec["factory"],
            "classification": spec["classification"],
            "paper_facing": spec["paper_facing"],
            "canonical_preset": spec["canonical_preset"],
            "comparison_role": spec["comparison_role"],
            "notes": spec["notes"],
            "algorithm_aliases": tuple(spec["algorithm_aliases"]),
            "algorithm": cfg.algorithm_family(),
            "quantizer_mode": cfg.quantizer_mode,
            "residual_mode": cfg.residual_mode,
            "rotation": cfg.rotation,
            "k_bits": cfg.k_bits,
            "k_group_size": cfg.k_group_size,
            "v_bits": cfg.v_bits,
            "v_group_size": cfg.v_group_size,
            "v_enabled": cfg.v_enabled,
            "qjl_proj_dim": cfg.qjl_proj_dim if cfg.residual_mode == "qjl" else None,
            "residual_topk": (
                cfg.residual_topk if cfg.residual_mode == "topk" else None
            ),
            "constructor": f'TurboQuantConfig.from_preset("{name}")',
        }

    @classmethod
    def preset_registry(cls) -> dict[str, dict[str, Any]]:
        return {name: cls.preset_metadata(name) for name in cls.preset_names()}

    @classmethod
    def from_preset(cls, name: str) -> TurboQuantConfig:
        if name not in _PRESET_REGISTRY:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(_PRESET_REGISTRY.keys())}"
            )

        spec = _PRESET_REGISTRY[name]
        factory = getattr(cls, str(spec["factory"]))
        kwargs = dict(spec["kwargs"])
        return cast(TurboQuantConfig, factory(**kwargs))

    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> TurboQuantConfig:
        residual_mode_kw = kwargs.get("residual_mode")
        residual_topk = kwargs.get("residual_topk", kwargs.get("residual", 0))
        if residual_mode_kw is None:
            residual_mode_kw = "qjl" if residual_topk == 0 else "topk"

        if residual_mode_kw == "topk":
            default_algorithm = "legacy_topk"
        elif residual_mode_kw == "none":
            default_algorithm = "paper_mse"
        else:
            default_algorithm = "paper_prod_qjl"

        cfg = cls(
            k_bits=kwargs.get("k_bits", 3),
            k_group_size=kwargs.get(
                "k_group_size",
                kwargs.get("group_size", 32),
            ),
            v_bits=kwargs.get("v_bits", 4),
            v_group_size=kwargs.get("v_group_size", 64),
            v_enabled=kwargs.get("v_enabled", True),
            v_scale_dtype=kwargs.get("v_scale_dtype", "float16"),
            rotation=kwargs.get("rotation", kwargs.get("rotation_mode", "hadamard")),
            rotation_seed=kwargs.get("rotation_seed", 1337),
            rotation_pad_to_pow2=bool(
                kwargs.get(
                    "rotation_pad_to_pow2",
                    kwargs.get("rotation_pad_to_pow2", True),
                )
            ),
            residual_mode=residual_mode_kw,
            residual_topk=residual_topk,
            resid_scale_bits=kwargs.get("resid_scale_bits", 8),
            scale_dtype=kwargs.get("scale_dtype", "float16"),
            eps=kwargs.get("eps", 1e-6),
            block_tokens=kwargs.get("block_tokens", 256),
            qjl_proj_dim=kwargs.get("qjl_proj_dim", 64),
            qjl_seed=kwargs.get("qjl_seed", 42),
            qjl_bits=kwargs.get("qjl_bits", 1),
            algorithm=kwargs.get("algorithm", default_algorithm),
            quantizer_mode=kwargs.get(
                "quantizer_mode",
                "polar"
                if cls.normalize_algorithm(kwargs.get("algorithm", default_algorithm))
                == "polarquant_exp"
                else "scalar",
            ),
            return_mode=kwargs.get("return_mode", "view"),
        )

        cfg.validate()
        return cfg

    def to_state_dict(self) -> dict:
        return {
            "k_bits": self.k_bits,
            "k_group_size": self.k_group_size,
            "v_bits": self.v_bits,
            "v_group_size": self.v_group_size,
            "v_enabled": self.v_enabled,
            "rotation": self.rotation,
            "rotation_seed": self.rotation_seed,
            "rotation_pad_to_pow2": self.rotation_pad_to_pow2,
            "residual_mode": self.residual_mode,
            "residual_topk": self.residual_topk,
            "resid_scale_bits": self.resid_scale_bits,
            "scale_dtype": self.scale_dtype,
            "v_scale_dtype": self.v_scale_dtype,
            "eps": self.eps,
            "block_tokens": self.block_tokens,
            "qjl_proj_dim": self.qjl_proj_dim,
            "qjl_seed": self.qjl_seed,
            "qjl_bits": self.qjl_bits,
            "quantizer_mode": self.quantizer_mode,
            "algorithm": self.algorithm_family(),
            "return_mode": self.return_mode,
        }
