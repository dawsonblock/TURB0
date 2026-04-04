"""PolarQuant — recursive polar transformation with zero-overhead quantisation.

Algorithm (arXiv:2502.02617, Zandieh et al., AISTATS 2026):
  1. Random preconditioning is already applied by rotation.py (Hadamard/Gaussian).
  2. Recursive polar transform (default L = 4 levels):
       Level 1:  pair (x[2j], x[2j+1]) → atan2(x[2j+1], x[2j]) ∈ [0, 2π),
                 radius = ‖(x[2j], x[2j+1])‖₂
       Level ℓ≥2: pair (r[2j], r[2j+1]) → atan2(r[2j+1], r[2j]) ∈ [0, π/2],
                  new_radius = ‖(r[2j], r[2j+1])‖₂
  3. Quantise angles:
       Level 1 → 4 bits (16 centroids, Lloyd-optimal on uniform [0, 2π))
       Levels 2..L → 2 bits (4 centroids each, Lloyd-optimal on the concentrated
                             distribution f(ψ) ∝ sin^{2^{ℓ-1}-1}(2ψ))
     Final d/2^L radii stored as float16 — no per-group scale factors.
  4. Memory for d=128, L=4: 496 bits / 128 dims ≈ 3.875 bits/dim.

References
----------
  Zandieh, Kacham, Karbasi, Mirrokni, Han (2026).
  "PolarQuant: Zero-overhead Quantization via Polar Decomposition."
  AISTATS 2026. arXiv:2502.02617.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters (paper Table 1)
# ─────────────────────────────────────────────────────────────────────────────

_N_LEVELS: int = 4  # L: recursive depth
_BITS_L1: int = 4   # bits for level-1 angles  (range [0, 2π))
_BITS_LE: int = 2   # bits for level-2+ angles (range [0, π/2))

# ─────────────────────────────────────────────────────────────────────────────
# Codebook construction
# ─────────────────────────────────────────────────────────────────────────────


def _lloyd_1d(
    samples: np.ndarray, n_centroids: int, n_iter: int = 200
) -> np.ndarray:
    """1-D Lloyd's algorithm (k-means) → sorted optimal centroids.

    Converges to locally optimal quantisation centroids for the empirical
    distribution given by *samples*.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples), n_centroids, replace=False)
    centroids = np.sort(samples[idx]).astype(np.float64)

    for _ in range(n_iter):
        dists = np.abs(samples[:, None] - centroids[None, :])  # [N, K]
        labels = np.argmin(dists, axis=1)
        new = np.array(
            [
                samples[labels == k].mean() if (labels == k).any() else centroids[k]
                for k in range(n_centroids)
            ],
            dtype=np.float64,
        )
        if np.max(np.abs(new - centroids)) < 1e-10:
            break
        centroids = np.sort(new)

    return centroids.astype(np.float32)


def _sample_level1_angles(n: int = 400_000) -> np.ndarray:
    """Level-1 angles are uniform on [0, 2π) after Gaussian preconditioning."""
    return np.random.default_rng(42).uniform(0.0, 2.0 * math.pi, n).astype(np.float32)


def _sample_level_ell_angles(level: int, n: int = 400_000) -> np.ndarray:
    """Draw samples from f(ψ) ∝ sin^{2^{ℓ-1}-1}(2ψ) on [0, π/2].

    Equivalent construction: ψ = atan2(‖y‖₂, ‖x‖₂) where x, y are
    independent N(0, I_{2^{ℓ-2}}) vectors (Lemma 3 of arXiv:2502.02617).
    """
    rng = np.random.default_rng(42 + level)
    half_dim = max(1, 1 << (level - 2))  # 2^(ℓ-2), minimum 1
    x = rng.standard_normal((n, half_dim)).astype(np.float32)
    y = rng.standard_normal((n, half_dim)).astype(np.float32)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    return np.arctan2(ny, nx).astype(np.float32)


def build_polar_codebooks(
    n_levels: int = _N_LEVELS,
    bits_l1: int = _BITS_L1,
    bits_le: int = _BITS_LE,
) -> list[np.ndarray]:
    """Build per-level quantisation codebooks via Lloyd's algorithm.

    Returns
    -------
    codebooks : list of n_levels numpy float32 arrays
        codebooks[0]   : level-1 centroids (bits_l1 bits, range [0, 2π))
        codebooks[i>0] : level-(i+1) centroids (bits_le bits, range [0, π/2))
    """
    codebooks: list[np.ndarray] = []

    # Level 1: uniform distribution on [0, 2π)
    codebooks.append(_lloyd_1d(_sample_level1_angles(), 1 << bits_l1))

    # Levels 2 .. n_levels: concentrated distribution near π/4
    for lv in range(2, n_levels + 1):
        codebooks.append(_lloyd_1d(_sample_level_ell_angles(lv), 1 << bits_le))

    return codebooks


_CB_CACHE: dict[tuple[int, int, int], list[np.ndarray]] = {}


def get_codebooks(
    n_levels: int = _N_LEVELS,
    bits_l1: int = _BITS_L1,
    bits_le: int = _BITS_LE,
) -> list[np.ndarray]:
    """Return cached codebooks, building them on first call."""
    key = (n_levels, bits_l1, bits_le)
    if key not in _CB_CACHE:
        _CB_CACHE[key] = build_polar_codebooks(n_levels, bits_l1, bits_le)
    return _CB_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Scalar quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _quantize_angles(angles: mx.array, codebook: np.ndarray) -> mx.array:
    """Nearest-centroid quantisation.

    Parameters
    ----------
    angles   : [..., m]  float32 angle values
    codebook : [K]       sorted float32 centroids (numpy)

    Returns
    -------
    codes : [..., m] uint8 codebook indices
    """
    cb = mx.array(codebook)                    # [K]
    diffs = mx.abs(angles[..., None] - cb)     # [..., m, K]
    return mx.argmin(diffs, axis=-1).astype(mx.uint8)


def _dequantize_angles(codes: mx.array, codebook: np.ndarray) -> mx.array:
    """Map uint8 indices to float32 codebook centroids.

    Parameters
    ----------
    codes    : [..., m]  uint8 codebook indices
    codebook : [K]       sorted float32 centroids (numpy)

    Returns
    -------
    angles : [..., m] float32
    """
    cb = mx.array(codebook)           # [K]
    return cb[codes.astype(mx.int32)] # [..., m]


# ─────────────────────────────────────────────────────────────────────────────
# Polar transform primitives
# ─────────────────────────────────────────────────────────────────────────────


def _polar_step_l1(x: mx.array) -> tuple[mx.array, mx.array]:
    """Level-1 polar step: pairs of real-valued coordinates.

    Parameters
    ----------
    x : [..., d]  (d must be even)

    Returns
    -------
    angles : [..., d//2]  in [0, 2π)
    radii  : [..., d//2]  ≥ 0
    """
    even = x[..., 0::2]   # "x" (cosine) coordinate of each pair
    odd  = x[..., 1::2]   # "y" (sine)   coordinate of each pair
    angles = mx.arctan2(odd, even)                               # (-π, π]
    angles = mx.where(angles < 0.0, angles + 2.0 * math.pi, angles)  # [0, 2π)
    radii  = mx.sqrt(even ** 2 + odd ** 2)
    return angles, radii


def _polar_step_ell(radii: mx.array) -> tuple[mx.array, mx.array]:
    """Level-ℓ (ℓ ≥ 2) polar step: pairs of non-negative radii.

    Parameters
    ----------
    radii : [..., m]  (m must be even, all values ≥ 0)

    Returns
    -------
    angles    : [..., m//2]  in [0, π/2]
    new_radii : [..., m//2]  ≥ 0
    """
    r_left  = radii[..., 0::2]
    r_right = radii[..., 1::2]
    # Both r_left, r_right ≥ 0 → atan2 result ∈ [0, π/2]
    angles    = mx.arctan2(r_right, r_left)
    new_radii = mx.sqrt(r_left ** 2 + r_right ** 2)
    return angles, new_radii


def polar_forward(
    x: mx.array, n_levels: int = _N_LEVELS
) -> tuple[list[mx.array], mx.array]:
    """Recursive polar decomposition.

    Parameters
    ----------
    x        : [..., d]  rotated key vectors; d must be divisible by 2^n_levels
    n_levels : int       recursion depth

    Returns
    -------
    angles_list : list of n_levels arrays
                  angles_list[ℓ] has shape [..., d // 2^{ℓ+1}]
    final_radii : [..., d // 2^n_levels]
    """
    angles_list: list[mx.array] = []
    a, r = _polar_step_l1(x)
    angles_list.append(a)                         # level 1

    for _ in range(n_levels - 1):
        a, r = _polar_step_ell(r)
        angles_list.append(a)                     # levels 2 .. n_levels

    return angles_list, r


def polar_inverse(
    angles_list: list[mx.array],
    final_radii: mx.array,
) -> mx.array:
    """Reconstruct x from the polar representation.

    Parameters
    ----------
    angles_list : list of n_levels arrays (level-1 first)
    final_radii : [..., d // 2^n_levels]

    Returns
    -------
    x : [..., d]
    """
    n_levels = len(angles_list)
    radii = final_radii  # [..., d/2^n_levels]

    # Reverse through levels n_levels → 2 (exclusive): rebuild radii of previous level
    for ell in range(n_levels - 1, 0, -1):
        angles = angles_list[ell]               # [..., m]
        left   = radii * mx.cos(angles)          # [..., m]
        right  = radii * mx.sin(angles)          # [..., m]
        # Interleave left/right to double the last dimension
        *pfx, m = radii.shape
        radii = mx.reshape(
            mx.stack([left, right], axis=-1),   # [..., m, 2]
            [*pfx, m * 2],                       # [..., 2m]
        )

    # Level-1 inverse → recover original x coordinates
    angles_l1 = angles_list[0]                  # [..., d//2]
    x_even = radii * mx.cos(angles_l1)           # [..., d//2]
    x_odd  = radii * mx.sin(angles_l1)           # [..., d//2]
    *pfx, m = x_even.shape
    return mx.reshape(
        mx.stack([x_even, x_odd], axis=-1),      # [..., d//2, 2]
        [*pfx, m * 2],                            # [..., d]
    )


# ─────────────────────────────────────────────────────────────────────────────
# PolarQuantPayload
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PolarQuantPayload:
    """Compressed key-block representation produced by PolarQuantizer.encode()."""

    angle_codes: list       # n_levels items; codes[ℓ]: [..., d//2^{ℓ+1}] uint8
    final_radii: mx.array   # [..., d//2^n_levels] float16
    d_orig:   int           # original d before padding
    d_pad:    int           # padded d (divisible by 2^n_levels)
    n_levels: int

    def byte_size(self) -> int:
        total = sum(
            int(code.nbytes)
            for code in self.angle_codes
            if hasattr(code, "nbytes")
        )
        if hasattr(self.final_radii, "nbytes"):
            total += int(self.final_radii.nbytes)
        return total

    def to_dict(self) -> dict[str, object]:
        def _arr_to_b64(arr) -> str:
            buf = io.BytesIO()
            np.save(buf, np.array(arr))
            return base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "angle_codes": [_arr_to_b64(code) for code in self.angle_codes],
            "final_radii": _arr_to_b64(self.final_radii),
            "d_orig": self.d_orig,
            "d_pad": self.d_pad,
            "n_levels": self.n_levels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PolarQuantPayload":
        def _b64_to_arr(b64: str):
            raw = base64.b64decode(b64.encode("ascii"))
            arr = np.load(io.BytesIO(raw))
            return mx.array(arr)

        angle_codes = [_b64_to_arr(code) for code in data.get("angle_codes", [])]
        final_radii = _b64_to_arr(data["final_radii"])
        return cls(
            angle_codes=angle_codes,
            final_radii=final_radii,
            d_orig=int(data.get("d_orig", 0)),
            d_pad=int(data.get("d_pad", 0)),
            n_levels=int(data.get("n_levels", len(angle_codes))),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PolarQuantizer
# ─────────────────────────────────────────────────────────────────────────────


class PolarQuantizer:
    """Zero-overhead KV-cache quantisation via recursive polar transformation.

    Implements arXiv:2502.02617 (Zandieh et al., AISTATS 2026).

    Parameters
    ----------
    n_levels : int
        Recursive depth.  Default 4.
    bits_l1  : int
        Quantisation bit-width for level-1 angles.  Default 4.
    bits_le  : int
        Quantisation bit-width for level-2+ angles.  Default 2.
    """

    def __init__(
        self,
        n_levels: int = _N_LEVELS,
        bits_l1: int = _BITS_L1,
        bits_le: int = _BITS_LE,
    ) -> None:
        self.n_levels = n_levels
        self.bits_l1 = bits_l1
        self.bits_le = bits_le
        # Build / retrieve cached numpy codebooks (built once per parameter set)
        self._codebooks_np: list[np.ndarray] = get_codebooks(n_levels, bits_l1, bits_le)

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(self, x: mx.array) -> PolarQuantPayload:
        """Compress x [..., d] → PolarQuantPayload.

        d is padded to the nearest multiple of 2^n_levels with zeros.
        """
        *prefix, d_orig = x.shape
        stride = 1 << self.n_levels          # 2^n_levels
        d_pad  = ((d_orig + stride - 1) // stride) * stride

        if d_pad != d_orig:
            pad = mx.zeros((*prefix, d_pad - d_orig), dtype=x.dtype)
            x = mx.concatenate([x, pad], axis=-1)

        angles_list, final_radii = polar_forward(x, self.n_levels)

        angle_codes: list[mx.array] = [
            _quantize_angles(a.astype(mx.float32), self._codebooks_np[i])
            for i, a in enumerate(angles_list)
        ]

        return PolarQuantPayload(
            angle_codes=angle_codes,
            final_radii=final_radii.astype(mx.float16),
            d_orig=d_orig,
            d_pad=d_pad,
            n_levels=self.n_levels,
        )

    # ── decode ────────────────────────────────────────────────────────────────

    def decode(self, payload: PolarQuantPayload) -> mx.array:
        """Reconstruct [..., d_orig] from a PolarQuantPayload."""
        angles_list = [
            _dequantize_angles(codes, self._codebooks_np[i]).astype(mx.float32)
            for i, codes in enumerate(payload.angle_codes)
        ]
        x_hat = polar_inverse(angles_list, payload.final_radii.astype(mx.float32))
        if payload.d_pad != payload.d_orig:
            x_hat = x_hat[..., : payload.d_orig]
        return x_hat

    # ── pipeline adapter (API-compatible with GroupScalarQuantizer) ────────────

    def quantize(
        self, x: mx.array, *, config=None
    ) -> tuple[PolarQuantPayload, None]:
        """Adapter matching the pipeline's ``quantize_main(x, *, config)`` signature.

        Returns ``(payload, None)`` — the ``None`` signals to pipeline.py that
        scales are absent and the PolarQuant path should be taken.
        """
        return self.encode(x), None

    def dequantize(
        self, payload: PolarQuantPayload, scales=None, *, config=None
    ) -> mx.array:
        """Adapter matching the pipeline's ``dequantize_main(p, s, *, config)`` signature."""
        return self.decode(payload)
