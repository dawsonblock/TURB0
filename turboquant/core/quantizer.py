"""
GroupScalarQuantizer — symmetric per-group N-bit scalar quantisation.

Encoding
--------
1. Reshape the last dimension into (n_groups, group_size).
2. Compute per-group absolute max → scale = max / q_max.
3. Round: code = clip(round(x / scale), −q_max, q_max).
4. Shift codes to unsigned: stored_code = code + q_max  → uint8/uint32.
5. Bit-pack stored_codes into uint32 words (cpw = 32 // bits codes per word).

All heavy work is vectorised MLX broadcast; no Python loops in the hot path.

Calibration
-----------
Call ``quantizer.fit(data)`` with a representative [N, D] sample before
inference.  This replaces per-batch dynamic scales with pre-computed per-group
statistics.  Without calibration, scales are computed dynamically per call.

API
---
    encode(x)          → packed: [..., n_words],  scales: [..., n_groups]
    decode(packed, scales, d) → [..., d]
    fit(data)          → calibrates static scales from [N, D] sample
"""

from __future__ import annotations

import mlx.core as mx

from turboquant.errors import TurboQuantShapeError

# ── Helpers ──────────────────────────────────────────────────────────────────


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _codes_per_word(bits: int) -> int:
    return 32 // bits


def _build_shifts(bits: int) -> mx.array:
    """Left-shift amounts for packing bits codes into one uint32."""
    cpw = _codes_per_word(bits)
    shifts = [bits * i for i in range(cpw)]
    return mx.array(shifts, dtype=mx.uint32)


def _build_mask(bits: int) -> mx.array:
    return mx.array((1 << bits) - 1, dtype=mx.uint32)


# ── Module-level shifted-mask cache so init cost is paid once ────────────────
_SHIFT_CACHE: dict[int, mx.array] = {}
_MASK_CACHE: dict[int, mx.array] = {}


def build_caches(bits: int) -> None:
    """Pre-allocate bit packing masks and shifts."""
    if bits not in _SHIFT_CACHE:
        _SHIFT_CACHE[bits] = _build_shifts(bits)
    if bits not in _MASK_CACHE:
        _MASK_CACHE[bits] = _build_mask(bits)


def _get_shifts(bits: int) -> mx.array:
    return _SHIFT_CACHE[bits]


def _get_mask(bits: int) -> mx.array:
    return _MASK_CACHE[bits]


# ── Pack / unpack ─────────────────────────────────────────────────────────────


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    """Pack uint32 codes [..., d_pad] → [..., n_words] (uint32).

    d_pad must be divisible by codes_per_word(bits).
    """
    cpw = _codes_per_word(bits)
    *prefix, d_pad = codes.shape
    if d_pad % cpw != 0:
        raise TurboQuantShapeError(f"d_pad={d_pad} not divisible by cpw={cpw}")
    n_words = d_pad // cpw

    words = codes.astype(mx.uint32).reshape(
        *prefix, n_words, cpw
    )  # [..., n_words, cpw]
    shifts = _get_shifts(bits)  # [cpw]
    return mx.sum(mx.left_shift(words, shifts), axis=-1)  # [..., n_words]


def unpack_codes(packed: mx.array, d_pad: int, bits: int) -> mx.array:
    """Unpack uint32 words [..., n_words] → uint32 codes [..., d_pad]."""
    cpw = _codes_per_word(bits)
    *prefix, n_words = packed.shape
    if n_words * cpw != d_pad:
        raise TurboQuantShapeError(
            f"Packed n_words={n_words} × cpw={cpw} = {n_words * cpw} ≠ d_pad={d_pad}"
        )
    mask = _get_mask(bits)
    shifts = _get_shifts(bits)  # [cpw]

    expanded = packed.reshape(*prefix, n_words, 1)  # [..., n_words, 1]
    # Broadcast: [..., n_words, cpw]
    codes = mx.bitwise_and(mx.right_shift(expanded, shifts), mask)
    return codes.reshape(*prefix, d_pad)  # [..., d_pad]


# ── Quantise / dequantise helpers ─────────────────────────────────────────────


def _compute_scales(
    x_groups: mx.array,
    q_max: int,
    eps: float,
    calibrated: mx.array | None,
) -> mx.array:
    """Return per-group scales.

    x_groups: [..., n_groups, group_size]
    calibrated: [n_groups] static scales, or None for dynamic.
    Returns: [..., n_groups, 1]
    """
    if calibrated is not None:
        n_groups = calibrated.shape[0]
        # Broadcast [n_groups] to [..., n_groups, 1]
        view_shape = (1,) * (x_groups.ndim - 2) + (n_groups, 1)
        return calibrated.reshape(*view_shape).astype(x_groups.dtype)

    abs_max = mx.max(mx.abs(x_groups), axis=-1, keepdims=True)
    return mx.maximum(abs_max / q_max, mx.array(eps, dtype=x_groups.dtype))


def quantize_groups(
    x: mx.array,
    bits: int,
    group_size: int,
    eps: float = 1e-6,
    calibrated_scales=None,
):
    """Quantise x [..., D] per group.

    Two-phase padding
    -----------------
    1. Pad to group boundary  (d_g  = ceil(D / group_size) * group_size).
    2. Pad to packing boundary (d_pack = ceil(d_g / cpw) * cpw).

    This handles all bit widths cleanly, including 3-bit where
    cpw = 32 // 3 = 10 does not divide typical head-dim values.

    Returns
    -------
    packed:  [..., n_words]   uint32
    scales:  [..., n_groups]  float (same dtype as x)
    """
    *prefix, d_orig = x.shape
    cpw = _codes_per_word(bits)
    q_max = (1 << (bits - 1)) - 1

    # Phase 1 — group padding
    n_groups = (d_orig + group_size - 1) // group_size
    d_g = n_groups * group_size

    if d_g > d_orig:
        z = mx.zeros((*x.shape[:-1], d_g - d_orig), dtype=x.dtype)
        x_g = mx.concatenate([x, z], axis=-1)
    else:
        x_g = x

    # Per-group scale computation
    xg = x_g.reshape(*prefix, n_groups, group_size)
    scales = _compute_scales(xg, q_max, eps, calibrated_scales)

    q = mx.round(xg / scales)
    q = mx.clip(q, -q_max, q_max)
    unsigned = (q.astype(mx.int32) + q_max).astype(mx.uint32)
    unsigned_flat = unsigned.reshape(*prefix, d_g)

    # Phase 2 — cpw packing padding
    d_pack = _round_up(d_g, cpw)
    if d_pack > d_g:
        z2 = mx.zeros((*unsigned_flat.shape[:-1], d_pack - d_g), dtype=mx.uint32)
        unsigned_flat = mx.concatenate([unsigned_flat, z2], axis=-1)

    packed = pack_codes(unsigned_flat, bits)
    return packed, scales.squeeze(-1)


_DEQUANT_CACHE = {}


def _inner_dequantize_groups(
    packed: mx.array,
    scales: mx.array,
    bits: int,
    group_size: int,
    d_orig: int,
    cpw: int,
    q_max: int,
    n_groups: int,
    d_pack: int,
    d_g: int,
) -> mx.array:
    unsigned = unpack_codes(packed, d_pack, bits)  # [..., d_pack]
    unsigned = unsigned[..., :d_g]  # crop to group region

    q = unsigned.astype(mx.int32) - q_max  # signed

    # Needs to handle arbitrary prefix shape for broadcast
    prefix = packed.shape[:-1]
    q_f = q.reshape(*prefix, n_groups, group_size).astype(scales.dtype)

    s = scales.reshape(*scales.shape[:-1], scales.shape[-1], 1)
    out = (q_f * s).reshape(*prefix, d_g)
    return out[..., :d_orig]


def dequantize_groups(
    packed: mx.array,
    scales: mx.array,
    bits: int,
    group_size: int,
    d_orig: int,
) -> mx.array:
    """Dequantise packed codes back to [..., d_orig]."""
    cpw = _codes_per_word(bits)
    q_max = (1 << (bits - 1)) - 1
    *prefix, n_words = packed.shape
    d_pack = n_words * cpw

    # n_groups comes from scales (authoritative), not from d_pack
    n_groups = scales.shape[-1]
    d_g = n_groups * group_size

    key = (bits, group_size, d_orig, cpw, q_max, n_groups, d_pack, d_g)
    if key not in _DEQUANT_CACHE:

        def fn(p, s):
            return _inner_dequantize_groups(p, s, *key)

        _DEQUANT_CACHE[key] = mx.compile(fn, shapeless=False)

    return _DEQUANT_CACHE[key](packed, scales)


# ── GroupScalarQuantizer ──────────────────────────────────────────────────────


class GroupScalarQuantizer:
    """Per-group symmetric N-bit scalar quantiser.

    Usage::

        q = GroupScalarQuantizer(n_bits=3, group_size=64)
        q.fit(calibration_data)   # optional; improves accuracy
        packed, scales = q.encode(x)
        x_hat = q.decode(packed, scales, x.shape[-1])
    """

    def __init__(
        self,
        n_bits: int = 3,
        group_size: int = 64,
        eps: float = 1e-6,
    ) -> None:
        if n_bits < 2 or n_bits > 8:
            raise TurboQuantShapeError(f"n_bits must be in [2,8], got {n_bits}")
        self.n_bits = n_bits
        self.group_size = group_size
        self.eps = eps
        build_caches(n_bits)
        self._calibrated_scales: mx.array | None = None

    # ── Calibration ──────────────────────────────────────────────────────────

    def fit(self, data: mx.array) -> None:
        """Calibrate per-group scales from a [N, D] representative sample.

        After calling ``fit``, ``encode`` uses static scales instead of
        computing them per-batch, which improves accuracy and throughput.
        """
        if data.ndim != 2:
            raise TurboQuantShapeError("Calibration data must be [N, D]")
        N, D = data.shape
        d_pad = _round_up(D, self.group_size)
        n_groups = d_pad // self.group_size
        q_max = (1 << (self.n_bits - 1)) - 1

        if d_pad > D:
            pad = mx.zeros((N, d_pad - D), dtype=data.dtype)
            data = mx.concatenate([data, pad], axis=-1)

        xg = data.reshape(N, n_groups, self.group_size)
        # EMA-stable estimate: mean of per-sample absolute maxima
        abs_max = mx.max(mx.abs(xg), axis=-1)  # [N, n_groups]
        mean_max = mx.mean(abs_max, axis=0)  # [n_groups]
        scales = mx.maximum(mean_max / q_max, mx.array(self.eps, dtype=data.dtype))
        mx.eval(scales)
        self._calibrated_scales = scales

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated_scales is not None

    def calibration_state(self) -> mx.array | None:
        """Return calibrated scales for serialization, if present."""
        return self._calibrated_scales

    def load_calibration_state(self, scales) -> None:
        """Restore calibrated scales from MLX or NumPy data."""
        self._calibrated_scales = None if scales is None else mx.array(scales)

    # ── Encode / decode ───────────────────────────────────────────────────────

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """x: [..., D] → (packed [..., n_words], scales [..., n_groups])."""
        d_orig = x.shape[-1]
        d_pad = _round_up(d_orig, self.group_size)

        if d_pad > d_orig:
            z = mx.zeros((*x.shape[:-1], d_pad - d_orig), dtype=x.dtype)
            x = mx.concatenate([x, z], axis=-1)

        packed, scales = quantize_groups(
            x, self.n_bits, self.group_size, self.eps, self._calibrated_scales
        )
        return packed, scales

    def decode(self, packed: mx.array, scales: mx.array, d_orig: int) -> mx.array:
        """(packed [..., n_words], scales [..., n_groups]) → [..., d_orig]."""
        return dequantize_groups(packed, scales, self.n_bits, self.group_size, d_orig)

    # ── Pipeline adapter wrappers ─────────────────────────────────────────────

    def quantize(self, x: mx.array, *, config=None) -> tuple[mx.array, mx.array]:
        """Adapter matching the ``quantize_main(x, *, config)`` pipeline signature.

        The *config* argument is accepted but ignored — all parameters
        are baked into this quantizer at construction time.
        """
        return self.encode(x)

    def dequantize(
        self, packed: mx.array, scales: mx.array, *, config=None
    ) -> mx.array:
        """Adapter matching the ``dequantize_main(packed, scales, *, config)`` signature.

        Decodes to the padded dimension (n_words × codes_per_word); callers
        are expected to slice to the original head dimension afterwards, which
        ``decode_k_block`` already does via ``block.d_rot``.
        """
        d_pad = int(packed.shape[-1]) * _codes_per_word(self.n_bits)
        return self.decode(packed, scales, d_pad)


# ── LloydMaxScalarQuantizer ───────────────────────────────────────────────────


class LloydMaxScalarQuantizer:
    """Lloyd-Max scalar quantizer with Beta(0.5, 0.5) (arcsine) centroid tables.

    Centroids are precomputed offline by Lloyd-Max iteration on the arcsine
    distribution on [-1, 1], which approximately describes rotated key-vector
    components after per-group max-absolute normalization (paper §3.1).

    Encoding
    --------
    1. Reshape last dim into (n_groups, group_size).
    2. Compute per-group max-abs scale.
    3. Normalise each group to [-1, 1].
    4. Assign each element to its nearest centroid (argmin squared distance).
    5. Bit-pack centroid indices into uint32 words.

    API is drop-in compatible with GroupScalarQuantizer.
    """

    # Offline Lloyd-Max centroids for arcsine distribution on [-1, 1].
    # Computed by 1000-iteration Lloyd-Max on a 2 M-point analytic grid;
    # enforced to be exactly symmetric around zero.
    # For b bits, there are 2^b centroids in ascending order.
    _LLOYDMAX_CENTROIDS: dict[int, list[float]] = {
        1: [-0.636620, 0.636620],
        2: [-0.855288, -0.297660, 0.297660, 0.855288],
        3: [
            -0.939694,
            -0.699434,
            -0.428660,
            -0.144242,
            0.144242,
            0.428660,
            0.699434,
            0.939694,
        ],
        4: [
            -0.974440,
            -0.870680,
            -0.751762,
            -0.624014,
            -0.490422,
            -0.352860,
            -0.212699,
            -0.071060,
            0.071060,
            0.212699,
            0.352860,
            0.490422,
            0.624014,
            0.751762,
            0.870680,
            0.974440,
        ],
    }

    def __init__(
        self,
        n_bits: int,
        group_size: int,
        eps: float = 1e-6,
    ) -> None:
        if n_bits not in self._LLOYDMAX_CENTROIDS:
            raise TurboQuantShapeError(
                f"LloydMaxScalarQuantizer: n_bits must be in {{1, 2, 3, 4}}, got {n_bits}"
            )
        self.n_bits = n_bits
        self.group_size = group_size
        self.eps = eps
        build_caches(n_bits)
        self._centroids_mx: mx.array = mx.array(
            self._LLOYDMAX_CENTROIDS[n_bits], dtype=mx.float32
        )

    # ── Encode / decode ───────────────────────────────────────────────────────

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """x: [..., D] → (packed [..., n_words], scales [..., n_groups])."""
        d_orig = int(x.shape[-1])
        d_pad = _round_up(d_orig, self.group_size)

        if d_pad > d_orig:
            z = mx.zeros((*x.shape[:-1], d_pad - d_orig), dtype=x.dtype)
            x = mx.concatenate([x, z], axis=-1)

        prefix = x.shape[:-1]
        n_groups = d_pad // self.group_size

        # Per-group max-abs scale (keeps normalised values in [-1, 1])
        xg = x.reshape(*prefix, n_groups, self.group_size)
        scales = mx.max(mx.abs(xg), axis=-1)  # [..., n_groups]
        scales = mx.maximum(scales, mx.array(self.eps, dtype=x.dtype))

        # Normalise and flatten
        x_norm = (xg / scales[..., None]).reshape(*prefix, d_pad)  # [..., d_pad]

        # Nearest centroid: [..., d_pad, 1] vs [n_centroids]
        c = self._centroids_mx.astype(x.dtype)
        diffs = x_norm[..., None] - c  # [..., d_pad, n_centroids]
        codes_idx = mx.argmin(diffs * diffs, axis=-1)  # [..., d_pad]

        # Bit-pack unsigned indices.
        # pad codes to next multiple of cpw (relevant for n_bits not a power-of-2,
        # e.g. n_bits=3 → cpw=10; d_pad=32 requires 40 codes to fill 4 words)
        cpw = _codes_per_word(self.n_bits)
        d_codes = int(codes_idx.shape[-1])
        d_pack = ((d_codes + cpw - 1) // cpw) * cpw
        if d_pack > d_codes:
            z = mx.zeros((*codes_idx.shape[:-1], d_pack - d_codes), dtype=mx.uint32)
            codes_idx = mx.concatenate([codes_idx, z], axis=-1)
        packed = pack_codes(codes_idx.astype(mx.uint32), self.n_bits)

        return packed, scales

    def decode(self, packed: mx.array, scales: mx.array, d_orig: int) -> mx.array:
        """(packed [..., n_words], scales [..., n_groups]) → [..., d_orig]."""
        prefix = packed.shape[:-1]
        n_words = int(packed.shape[-1])
        n_groups = int(scales.shape[-1])
        d_pack = n_words * _codes_per_word(self.n_bits)
        d_g = n_groups * self.group_size

        # Unpack to unsigned centroid indices
        unsigned = unpack_codes(packed, d_pack, self.n_bits)  # [..., d_pack]

        # Centroid lookup and scale back
        c = self._centroids_mx.astype(scales.dtype)
        x_norm = c[unsigned][..., :d_g].reshape(*prefix, n_groups, self.group_size)
        x_hat = (x_norm * scales.reshape(*prefix, n_groups, 1)).reshape(*prefix, d_g)

        return x_hat[..., :d_orig]

    # ── Pipeline adapter wrappers ─────────────────────────────────────────────

    def quantize(self, x: mx.array, *, config=None) -> tuple[mx.array, mx.array]:
        """Adapter matching the ``quantize_main(x, *, config)`` pipeline signature."""
        return self.encode(x)

    def dequantize(
        self, packed: mx.array, scales: mx.array, *, config=None
    ) -> mx.array:
        """Adapter matching the ``dequantize_main(packed, scales, *, config)`` signature.

        Decodes to the padded dimension; callers slice to the original head
        dimension afterwards (``decode_k_block`` does this via ``block.d_rot``).
        """
        d_pad = int(packed.shape[-1]) * _codes_per_word(self.n_bits)
        return self.decode(packed, scales, d_pad)
