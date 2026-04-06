"""
PolarQuant vs GroupScalarQuantizer benchmark.

Measures encode latency, decode latency, and reconstruction MSE for both
quantisers across typical KV-cache head dimensions and sequence lengths.

Usage::

    python benchmarks/exploratory/bench_polar_vs_scalar.py

Writes a table to stdout and a JSON summary to artifacts/benchmarks/polar_vs_scalar.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)

import mlx.core as mx
import numpy as np

from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.quantizer import GroupScalarQuantizer

# ── Config ────────────────────────────────────────────────────────────────────

HEAD_DIMS = [64, 128, 256]
SEQ_LENS = [256, 512, 1024]
N_HEADS = 8
BATCH = 1
REPS = 50  # timing repetitions
WARMUP = 10
SCALAR_BITS = 3
SCALAR_GRP = 64

# ── Helpers ───────────────────────────────────────────────────────────────────


def _time_fn(fn, *args, reps=REPS, warmup=WARMUP):
    """Return (mean_ms, std_ms) for calling fn(*args)."""
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, tuple):
            for o in out:
                if isinstance(o, mx.array):
                    mx.eval(o)

    times_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, tuple):
            for o in out:
                if isinstance(o, mx.array):
                    mx.eval(o)
        times_ms.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(times_ms)
    return float(arr.mean()), float(arr.std())


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a - b) ** 2))


# ── Benchmark ─────────────────────────────────────────────────────────────────


def run(head_dim: int, seq_len: int) -> dict:
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((BATCH, N_HEADS, seq_len, head_dim)).astype(np.float32)
    x = mx.array(x_np)
    mx.eval(x)

    # Flatten to [B*NH, T, D] for quantizers that operate on last dim
    x_flat = x.reshape(BATCH * N_HEADS * seq_len, head_dim)
    mx.eval(x_flat)

    result: dict = {"head_dim": head_dim, "seq_len": seq_len}

    # ── Scalar quant ─────────────────────────────────────────────────────────
    sq = GroupScalarQuantizer(n_bits=SCALAR_BITS, group_size=SCALAR_GRP)

    def sq_encode():
        return sq.encode(x_flat)

    enc_mean, enc_std = _time_fn(sq_encode)
    packed_s, scales_s = sq_encode()
    mx.eval(packed_s, scales_s)

    def sq_decode():
        return sq.decode(packed_s, scales_s, head_dim)

    dec_mean, dec_std = _time_fn(sq_decode)
    x_sq = sq.decode(packed_s, scales_s, head_dim)
    mse_scalar = _mse(x_flat, x_sq)

    # Scalar bits/dim: n_bits for packed data + 16 bits per scale element
    n_groups = -(-head_dim // SCALAR_GRP)  # ceil div
    bits_scalar = (
        SCALAR_BITS * head_dim + 16 * n_groups
    ) / head_dim  # effective bits/dim

    result["scalar"] = {
        "encode_ms_mean": round(enc_mean, 3),
        "encode_ms_std": round(enc_std, 3),
        "decode_ms_mean": round(dec_mean, 3),
        "decode_ms_std": round(dec_std, 3),
        "mse": round(float(mse_scalar), 6),
        "bits_per_dim": round(bits_scalar, 3),
    }

    # ── PolarQuant ────────────────────────────────────────────────────────────
    pq = PolarQuantizer()  # n_levels=4, bits_l1=4, bits_le=2

    def pq_encode():
        return pq.encode(x_flat)

    enc_mean_p, enc_std_p = _time_fn(pq_encode)
    payload = pq_encode()
    for c in payload.angle_codes:
        mx.eval(c)
    mx.eval(payload.final_radii)

    def pq_decode():
        return pq.decode(payload)

    dec_mean_p, dec_std_p = _time_fn(pq_decode)
    x_pq = pq.decode(payload)
    mse_polar = _mse(x_flat, x_pq)

    # PolarQuant bits/dim: 4 bits L1 + 2 bits x (L-1) levels + 16 bits
    # for d/2^L final radii → formula from paper §4
    n_l = pq.n_levels
    d_pad = (-(-head_dim // (1 << n_l))) * (1 << n_l)
    total_angle_bits = pq.bits_l1 * (d_pad // 2) + pq.bits_le * sum(
        d_pad // (2**lv) for lv in range(2, n_l + 1)
    )
    total_radii_bits = 16 * (d_pad // (2**n_l))
    bits_polar = (total_angle_bits + total_radii_bits) / head_dim

    result["polar"] = {
        "encode_ms_mean": round(enc_mean_p, 3),
        "encode_ms_std": round(enc_std_p, 3),
        "decode_ms_mean": round(dec_mean_p, 3),
        "decode_ms_std": round(dec_std_p, 3),
        "mse": round(float(mse_polar), 6),
        "bits_per_dim": round(bits_polar, 3),
    }

    return result


# ── Entry ─────────────────────────────────────────────────────────────────────


def main():
    print(f"\n{'=' * 90}")
    print(f"  PolarQuant vs GroupScalarQuantizer({SCALAR_BITS}-bit, g={SCALAR_GRP})")
    print(f"  batch={BATCH}  n_heads={N_HEADS}  reps={REPS}  warmup={WARMUP}")
    print(f"{'=' * 90}")

    hdr = (
        f"{'d_head':>7} {'T':>6}  "
        f"{'scalar enc ms':>14} {'scalar dec ms':>14} {'scalar MSE':>12} {'scalar b/d':>10}  "
        f"{'polar enc ms':>14} {'polar dec ms':>14} {'polar MSE':>12} {'polar b/d':>10}  "
        f"{'MSE ratio':>10}"
    )
    print(hdr)
    print("-" * 90)

    all_results = []
    for d in HEAD_DIMS:
        for t in SEQ_LENS:
            r = run(d, t)
            s = r["scalar"]
            p = r["polar"]
            mse_ratio = p["mse"] / s["mse"] if s["mse"] > 0 else float("nan")
            print(
                f"{d:>7} {t:>6}  "
                f"{s['encode_ms_mean']:>10.3f}±{s['encode_ms_std']:<6.3f}"
                f" {s['decode_ms_mean']:>10.3f}±{s['decode_ms_std']:<6.3f}"
                f" {s['mse']:>12.6f} {s['bits_per_dim']:>10.3f}  "
                f"{p['encode_ms_mean']:>10.3f}±{p['encode_ms_std']:<6.3f}"
                f" {p['decode_ms_mean']:>10.3f}±{p['decode_ms_std']:<6.3f}"
                f" {p['mse']:>12.6f} {p['bits_per_dim']:>10.3f}  "
                f"{mse_ratio:>10.2f}x"
            )
            all_results.append({**r, "mse_ratio": round(mse_ratio, 4)})

    print("=" * 90)

    out_dir = os.path.join(_ROOT, "artifacts", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "polar_vs_scalar.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON written → {out_path}")


if __name__ == "__main__":
    main()
