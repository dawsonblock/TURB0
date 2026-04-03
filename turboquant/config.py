from __future__ import annotations

from dataclasses import dataclass


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
    block_tokens: int = 256

    qjl_proj_dim: int = 64
    qjl_seed: int = 42
    qjl_bits: int = 1

    quantizer_mode: str = "scalar"   # "scalar" | "polar"
    algorithm: str = "turboquant_prod"  # "turboquant_mse" | "turboquant_prod"
    return_mode: str = "view"

    # ------------------------------------------------------------------ #
    # Algorithm mode helpers                                               #
    # ------------------------------------------------------------------ #

    def is_mse_mode(self) -> bool:
        """True when only scalar quantization is used (no QJL residual)."""
        return self.algorithm == "turboquant_mse"

    def is_prod_mode(self) -> bool:
        """True when 1-bit QJL residual is used for inner-product estimation."""
        return self.algorithm == "turboquant_prod"

    # ------------------------------------------------------------------ #
    # Effective bits-per-channel formulae (paper §3)                      #
    # ------------------------------------------------------------------ #

    def effective_bits_per_channel_k(self, d: int) -> float:
        """Effective bits/channel for K cache at head-dim *d*.

        MSE:  b + 16 / g                (scalar quant + fp16 scale per group)
        Prod: (b-1) + 16/g + (p+16)/d  (one fewer quant bit, plus QJL bits
                                         and its fp16 norm stored per dim)
        where b = k_bits, g = k_group_size, p = qjl_proj_dim.
        """
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
        return (self.effective_bits_per_channel_k(d) +
                self.effective_bits_per_channel_v(d)) / 2.0

    def validate(self) -> None:
        if self.algorithm not in {"turboquant_mse", "turboquant_prod"}:
            raise ValueError(
                f"algorithm must be 'turboquant_mse' or 'turboquant_prod', "
                f"got {self.algorithm!r}"
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

        if self.residual_mode == "topk" and self.residual_topk <= 0:
            raise ValueError(
                "residual_topk must be > 0 when residual_mode='topk'"
            )

        if self.residual_mode == "qjl":
            if self.qjl_bits != 1:
                raise ValueError(
                    f"Only 1-bit QJL is currently supported, got {self.qjl_bits}"
                )
            if self.qjl_proj_dim <= 0:
                raise ValueError(f"qjl_proj_dim must be > 0, got {self.qjl_proj_dim}")

        if self.quantizer_mode not in {"scalar", "polar"}:
            raise ValueError(f"Unsupported quantizer_mode: {self.quantizer_mode!r}"
                             "; expected 'scalar' or 'polar'")

        # Algorithm-residual_mode contract (paper §3)
        if self.algorithm == "turboquant_mse" and self.residual_mode != "none":
            raise ValueError(
                "algorithm='turboquant_mse' requires residual_mode='none', "
                f"got residual_mode={self.residual_mode!r}"
            )
        if self.algorithm == "turboquant_prod" and self.residual_mode == "none":
            raise ValueError(
                "algorithm='turboquant_prod' requires a residual encoder "
                "(residual_mode='qjl'; 'topk' is accepted as experimental). "
                f"Got residual_mode='none'."
            )

    @classmethod
    def from_preset(cls, name: str) -> TurboQuantConfig:
        """
        Return a configuration for a named optimization preset.

        Presets
        -------
        - "paper_mse"       Paper §3 MSE stage: rotate + Lloyd-Max scalar quant, no residual.
                            Effective ~3.5 bits/channel at k_bits=3, k_group_size=64.
        - "paper_prod"      Paper §3 Prod stage: MSE stage (k_bits=2) + 1-bit QJL residual.
                            Unbiased inner-product estimation in rotated space.
        - "high_compression" (3-bit K, 4-bit V, QJL residuals, 64-group) [legacy]
        - "balanced"        (4-bit K, 4-bit V, Top-2 residuals, 32-group) [legacy]
        - "max_quality"     (4-bit K, 8-bit V, Top-4 residuals, 16-group) [legacy]
        """
        presets = {
            # ---- Paper-faithful presets ----
            "paper_mse": cls(
                algorithm="turboquant_mse",
                k_bits=3, k_group_size=64,
                v_bits=4, v_group_size=64,
                rotation="hadamard",
                residual_mode="none",
            ),
            "paper_prod": cls(
                algorithm="turboquant_prod",
                k_bits=3, k_group_size=64,
                v_bits=4, v_group_size=64,
                rotation="hadamard",
                residual_mode="qjl", qjl_bits=1, qjl_proj_dim=64,
            ),
            # ---- Legacy presets (kept for backward compat) ----
            "high_compression": cls(
                algorithm="turboquant_prod",
                k_bits=3, k_group_size=64,
                v_bits=4, v_group_size=64,
                residual_mode="qjl", qjl_bits=1,
                rotation="hadamard"
            ),
            "balanced": cls(
                algorithm="turboquant_prod",
                k_bits=4, k_group_size=32,
                v_bits=4, v_group_size=32,
                residual_mode="qjl", qjl_bits=1,
                rotation="hadamard"
            ),
            "max_quality": cls(
                algorithm="turboquant_prod",
                k_bits=4, k_group_size=16,
                v_bits=8, v_group_size=16,
                residual_mode="qjl", qjl_bits=1,
                rotation="hadamard"
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        return presets[name]

    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> TurboQuantConfig:
        """
        Thin migration shim for older callers.
        """
        residual_mode_kw = kwargs.get("residual_mode")
        residual_topk = kwargs.get("residual_topk", kwargs.get("residual", 0))
        # Infer residual_mode when not explicit
        if residual_mode_kw is None:
            residual_mode_kw = "qjl" if residual_topk == 0 else "topk"
        # Infer algorithm from residual_mode.
        # Both "qjl" and "topk" are residual encoders → turboquant_prod.
        # "none" means pure MSE (no residual) → turboquant_mse.
        if residual_mode_kw in ("qjl", "topk"):
            default_algorithm = "turboquant_prod"
        else:
            default_algorithm = "turboquant_mse"

        cfg = cls(
            k_bits=kwargs.get("k_bits", kwargs.get("k_bits", 3)),
            k_group_size=kwargs.get("k_group_size", kwargs.get("group_size", 32)),
            v_bits=kwargs.get("v_bits", 4),
            v_group_size=kwargs.get("v_group_size", 64),
            v_enabled=kwargs.get("v_enabled", True),
            v_scale_dtype=kwargs.get("v_scale_dtype", "float16"),
            rotation=kwargs.get("rotation", kwargs.get("rotation_mode", "hadamard")),
            rotation_seed=kwargs.get("rotation_seed", 1337),
            rotation_pad_to_pow2=bool(kwargs.get("rotation_pad_to_pow2", kwargs.get("rotation_pad_to_por", True))),
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
            "algorithm": self.algorithm,
            "return_mode": self.return_mode,
        }
